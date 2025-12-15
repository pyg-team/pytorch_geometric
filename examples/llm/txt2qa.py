import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import traceback
from copy import deepcopy
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from torch_geometric.llm.models.qa_gen import (
    LLMClient,
    QAGenerationState,
    TaskStatus,
    compute_metrics,
    convert_posixpath_to_string,
    create_artifacts,
    end,
    evaluate_qa_pairs,
    feedback_controller,
    filter_by_quality,
    generate_negative_answers_node,
    generate_qa_pairs,
    get_text_artifacts,
    load_input,
    mine_hard_negative_segments,
    monitor_progress,
    route_after_validation,
    should_retry,
    store_results,
    validate_qa_pairs,
)

logger = logging.getLogger(__name__)


class QAGenerator:
    def __init__(
        self,
        config_path: str,
        overwrite: bool = False,
        batch: int = 0,
        batch_size: int = 10,
        backend: str = 'nim',
        gen_model: str = 'nvdev/meta/llama-3.1-70b-instruct',
        embedding_model: str = 'nvdev/nvidia/llama-3.2-nv-embedqa-1b-v2',
        evaluation_model: str = 'nvdev/meta/llama-3.1-70b-instruct',
        base_url: str = None,
        api_key: str = None,
        vllm_tensor_parallel_size: int = 1,
        vllm_gpu_memory_utilization: float = 0.9,
        vllm_max_model_len: Optional[int] = None,
        model_progression: List[str] = None,
    ):
        self.config = self._load_config(config_path)
        self.workflow = self._build_qa_generation_workflow()
        self.active_tasks = {}
        self.overwrite = overwrite
        self.batch = batch
        self.batch_size = batch_size
        self.backend = backend
        self.gen_model = (gen_model if not self.config.get('gen_model', None)
                          else self.config.get('gen_model'))
        self.embedding_model = (embedding_model
                                if not self.config.get('embedding_model', None)
                                else self.config.get('embedding_model'))

        evaluation_obj = self.config.get('evaluation', {})
        self.evaluation_model = (evaluation_model
                                 if not evaluation_obj.get('model', None) else
                                 evaluation_obj.get('model'))

        model_progression = self.config.get('model_progression', None)
        if model_progression is None:
            model_progression = [
                'nvdev/mistralai/mixtral-8x22b-instruct-v0.1',
                'nvdev/meta/llama-3.1-70b-instruct',
                'nvdev/meta/llama-3.1-405b-instruct',
                'nvdev/nvidia/llama-3.1-nemotron-ultra-253b-v1',
            ]
        self.model_progression = model_progression
        self.base_url = base_url

        # Initialize client based on backend
        self.client = LLMClient(
            generation_model=self.gen_model,
            embedding_model=self.embedding_model,
            evaluation_model=self.evaluation_model,
            backend=self.backend,
            api_key=api_key,
            tensor_parallel_size=vllm_tensor_parallel_size,
            gpu_memory_utilization=vllm_gpu_memory_utilization,
            max_model_len=vllm_max_model_len,
            enable_sleep_mode=True,
        )

    def _build_qa_generation_workflow(self) -> CompiledStateGraph:
        """Build the QA generation workflow using LangGraph."""
        # Create the graph
        workflow = StateGraph(QAGenerationState)

        # Add nodes
        workflow.add_node('load_input', load_input)
        workflow.add_node('create_artifacts', create_artifacts)
        workflow.add_node('generate_qa', generate_qa_pairs)
        workflow.add_node('validate', validate_qa_pairs)
        workflow.add_node('mine_hard_negative_segments',
                          mine_hard_negative_segments)
        workflow.add_node('generate_negative', generate_negative_answers_node)
        workflow.add_node('evaluate_qa',
                          evaluate_qa_pairs)  # New evaluation node
        workflow.add_node('filter_quality', filter_by_quality)
        workflow.add_node('monitor', monitor_progress)
        workflow.add_node('store_results', store_results)
        workflow.add_node('feedback_node', feedback_controller)
        workflow.add_node('end', end)

        # Add edges
        workflow.set_entry_point('load_input')
        workflow.add_edge('load_input', 'create_artifacts')
        workflow.add_edge('create_artifacts', 'generate_qa')
        workflow.add_edge('generate_qa', 'validate')

        # Conditional edges
        workflow.add_conditional_edges(
            'validate',
            route_after_validation,
            {
                'continue':
                'mine_hard_negative_segments',  # Continue to mine hard negatives
                'feedback': 'feedback_node',  # Route to feedback on failure
            },
        )

        workflow.add_edge(
            'mine_hard_negative_segments',
            'generate_negative')  # Then generate negative answers
        workflow.add_edge('generate_negative',
                          'evaluate_qa')  # Add evaluation after negatives
        workflow.add_edge('evaluate_qa', 'filter_quality')
        workflow.add_edge('filter_quality', 'monitor')
        workflow.add_edge('monitor', 'store_results')

        workflow.add_conditional_edges(
            'store_results',
            lambda x:
            ('feedback'
             if x.get('status') == TaskStatus.FAILED.value else 'end'),
            {
                'feedback_node': 'feedback_node',
                'end': 'end'
            },
        )

        workflow.add_conditional_edges(
            'feedback_node',
            should_retry,
            {
                # "continue": None,
                'feedback_node': 'feedback_node',
            },
        )

        return workflow.compile()

    def _load_config(self, config_path: str) -> Dict:
        with open(config_path) as f:
            return yaml.safe_load(f)

    def check_already_done(self):
        if self.overwrite:
            return []
        output_dir = Path(self.config.get('output_dir'))
        all_qa_files = glob(f'{output_dir}/all_qa_pairs_batch_*.jsonl')
        logger.info('How many all_qa files? %d', len(all_qa_files))
        done_file_paths = []
        for f in all_qa_files:
            with open(f) as fin:
                for line in fin:
                    data = json.loads(line.strip())
                    qa_pairs = data['qa_pairs']
                    for qa_pair in qa_pairs:
                        done_file_paths.append(qa_pair['file_path'])
        done_file_paths = list(set(done_file_paths))
        done_file_paths.sort()
        logger.info('How many files already done? %d', len(done_file_paths))
        return done_file_paths

    async def process_directory(self):
        input_dir = self.config.get('input_dir', None)
        folder = None
        if input_dir:
            folder = Path(input_dir)
            if not folder.exists():
                logger.error('Folder not found: %s', input_dir)
                return

        done_file_paths = self.check_already_done()
        output_dir = self.config.get('output_dir')

        input_data = []
        # Use rglob to recursively search in subfolders for common text file extensions
        text_extensions = ['*.txt', '*.text', '*.md']
        for ext in text_extensions:
            input_data.extend(folder.rglob(ext))

        # Also include files without any extension (treat as text files)
        all_files = folder.rglob('*')
        input_data.extend(
            [f for f in all_files if f.is_file() and not f.suffix])

        logger.info('total input files: %d', len(input_data))

        limiter = 5
        batch_size = self.batch_size
        batch = self.batch
        input_data.sort()

        start_idx = batch * batch_size
        end_idx = (batch + 1) * batch_size
        if end_idx > len(input_data):
            end_idx = len(input_data)
        logger.info('input files from %d to %d', start_idx, end_idx)
        input_data = input_data[start_idx:end_idx]

        # Collect results from all chunks
        all_chunk_results = []
        start = 0
        output_dir = Path(self.config.get('output_dir'))
        output_dir.mkdir(exist_ok=True)
        out_file = f'{output_dir}/all_qa_pairs_batch_{batch}.jsonl'
        # Check if file exists before removing
        if os.path.exists(out_file) and self.overwrite:
            os.remove(out_file)

        while True:
            end = (start + limiter if start +
                   limiter < len(input_data) else len(input_data))
            logger.info('Process %d to %d', start, end)
            total_input_data = input_data[start:end]

            input_data_to_do = [
                elem for elem in total_input_data
                if str(elem) not in done_file_paths
            ]

            logger.info(
                f'Already done: {len(done_file_paths)}, still_to_do: {len(input_data_to_do)}'
            )

            if len(input_data_to_do) > 0:
                results = await self._process_directory(
                    input_data_to_do, folder)
                # incremental save
                logger.info('results: %d', len(results))
                self.write_all_qa_pairs(results, out_file)
                all_chunk_results.extend(results)
                logger.info('Done from %d to %d', start, end)
            else:
                logger.info('All processed.')
            start = end

            if start >= len(input_data):
                break

        # sanity check
        if not os.path.exists(out_file):
            logger.warning('%s not exist', out_file)
            lines = []
        else:
            with open(out_file) as fin:
                lines = fin.readlines()
        if len(lines) != len(all_chunk_results):
            logger.debug('%s: num lines: %d', out_file, len(lines))
            logger.debug('all_chunk_results: %d', len(all_chunk_results))
            logger.warning(
                f'{out_file} contains {len(lines)} json objs, but there are {len(all_chunk_results)} all_chunk_results'
            )

    async def _process_directory(self, input_data_list: List[Any],
                                 input_dir: Optional[Path] = None):
        """Process all transcription files in a directory."""
        # input_data can be a list of Path (filename)
        tasks = []

        self.config.get('output_dir')
        for input_data in input_data_list:
            logger.info('==> input_file: %s', input_data)
            data = input_data
            data_id = input_data
            task = asyncio.create_task(
                self.process_data(data, data_id, input_dir))
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Per-chunk summary statistics
        successful = sum(
            1 for r in results
            if isinstance(r, dict) and r.get('status') == TaskStatus.COMPLETED.
            value and r.get('reason') != 'already_processed')
        skipped = sum(
            1 for r in results
            if isinstance(r, dict) and r.get('reason') == 'already_processed')
        failed = sum(1 for r in results if isinstance(r, dict)
                     and r.get('status') == TaskStatus.FAILED.value)

        if self.overwrite:
            logger.info(
                f'Chunk processing complete (overwrite mode): {successful} successful, {failed} failed'
            )
        else:
            logger.info(
                f'Chunk processing complete: {successful} successful, {skipped} skipped (already processed), {failed} failed'
            )

        return results

    @staticmethod
    def _get_output_stem(file_path: Path, input_dir: Path) -> str:
        """Generate a sanitized output filename stem that includes subfolder paths.

        For files in subfolders, concatenates subfolder names with the filename stem.
        Example: input_dir/subfolder1/subfolder2/file.txt -> subfolder1_subfolder2_file

        Args:
            file_path: Full path to the input file
            input_dir: Base input directory

        Returns:
            Sanitized stem string suitable for use in output filenames
        """
        try:
            # Get the relative path from input_dir to file_path
            relative_path = file_path.relative_to(input_dir)

            # Get all parts (subfolders + filename stem)
            parts = list(relative_path.parent.parts) + [relative_path.stem]

            # Filter out empty parts and join with underscore
            sanitized_parts = [part for part in parts if part and part != '.']
            output_stem = '_'.join(sanitized_parts)

            # Remove any problematic characters for filenames
            output_stem = re.sub(r'[^\w\-_]', '_', output_stem)

            return output_stem
        except ValueError:
            # If file_path is not relative to input_dir, just use the stem
            logger.warning(
                f'File {file_path} is not relative to input_dir {input_dir}, using stem only'
            )
            return file_path.stem

    async def process_data(self, data: str, data_id: str,
                           input_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Process a single transcription file."""
        # data can be a file_path or an image_b64_str
        # use md5 hash of the file name (without extension) to retrieve the output file
        if os.path.exists(data):
            file_path = data
            task_id = hashlib.md5(Path(file_path).stem.encode()).hexdigest()
            is_image_b64_str = False
        else:
            file_path = data_id
            task_id = data_id
            is_image_b64_str = True

        # Check if output file already exists (resume functionality)
        # Skip this check if --overwrite flag is used
        if not self.overwrite:
            output_dir = Path(self.config.get('output_dir'))
            # Use stem to be consistent with the output file name in store_results
            if os.path.exists(data):
                # Use the new helper to generate output stem with subfolder paths
                if input_dir:
                    stem = self._get_output_stem(Path(file_path), input_dir)
                else:
                    stem = Path(file_path).stem
            else:
                stem = data_id
            output_file = output_dir / f'qa_pairs_{stem}.json'

            if output_file.exists():
                logger.info(
                    f'Skipping {file_path} - output file already exists: {output_file}'
                )
                return {
                    'task_id': task_id,
                    'status': TaskStatus.COMPLETED.value,
                    'reason': 'already_processed',
                    'file_path': file_path,
                    'is_image_b64_str': is_image_b64_str,
                    'output_file': str(output_file),
                }
        else:
            logger.info(
                f'Overwrite mode enabled - processing {file_path} regardless of existing output files'
            )

        try:
            initial_state = {
                'task_id':
                task_id,
                'file_path':
                file_path,
                'input_dir':
                str(input_dir) if input_dir else None,
                'output_dir':
                self.config.get('output_dir'),
                'qa_pairs': [],
                'question_only':
                self.config.get('question_only', False),
                'validation_results': {},
                'negative_answers': [],
                'model_selection':
                '',
                'retry_count':
                0,
                'error_messages': [],
                'status':
                TaskStatus.PENDING.value,
                'metrics': {},
                'feedback': {},
                # "client": self.client,  # Pass OpenAI client
                'num_pairs':
                self.config.get('num_pairs', 10),
                'num_negatives':
                self.config.get('num_negatives', 3),
                'dedup_threshold':
                self.config.get('dedup_threshold', 0.6),
                'parts':
                self.config.get('parts', 3),
                'hard':
                self.config.get('hard', False),
                'use_artifact':
                self.config.get('use_artifact', False),
                'evaluation':
                self.config.get('evaluation'),
                'query_type_distribution':
                self.config.get(
                    'query_type_distribution',
                    '{"multi_hop":0.4,"structural":0.3,"contextual":0.3}',
                ),
                'reasoning_type_distribution':
                self.config.get('reasoning_type_distribution', None),
                'min_hops':
                self.config.get('min_hops', 2),
                'max_hops':
                self.config.get('max_hops', 4),
                'models':
                self.config.get('models'),
                'summary': [],
                'self_contained_question':
                self.config.get('self_contained_question', False),
                'hard_negatives_top_k':
                self.config.get('hard_negatives_top_k', 10),
                'hard_negatives_min_sim':
                self.config.get('hard_negatives_min_sim', 0.5),
                'hard_negatives_max_sim':
                self.config.get('hard_negatives_max_sim', 0.7),
                'enable_hard_negatives':
                self.config.get('enable_hard_negatives', True),
                'text_artifacts':
                None,
            }

            use_artifact = initial_state.get('use_artifact')
            if use_artifact:
                try:
                    ret = await load_input(initial_state)
                    text_artifacts = get_text_artifacts(
                        ret['segments'], self.client)
                    logger.info('text artifacts: %d', len(text_artifacts))
                    initial_state['text_artifacts'] = text_artifacts
                except Exception as e:
                    logger.warning('Failed to retrieve text artifacts: %s', e)

            total_pairs = initial_state['num_pairs']
            pairs_per_shard = int(self.config.get('pairs_per_shard', 0) or 0)
            max_parallel_shards = int(
                self.config.get('max_parallel_shards', 4) or 1)

            # If sharding requested and beneficial, fan out multiple workflow runs
            if pairs_per_shard > 0 and pairs_per_shard < total_pairs:
                logger.info(
                    f'Sharding task {task_id}: total_pairs={total_pairs}, '
                    f'pairs_per_shard={pairs_per_shard}, max_parallel_shards={max_parallel_shards}'
                )

                shard_states: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
                remaining = total_pairs
                shard_idx = 0

                while remaining > 0:
                    request_pairs = min(pairs_per_shard, remaining)
                    shard_state = deepcopy(initial_state)
                    shard_state['client'] = self.client
                    shard_task_id = f'{task_id}-shard-{shard_idx}'
                    shard_state['task_id'] = shard_task_id
                    shard_state['num_pairs'] = request_pairs
                    shard_state['parent_task_id'] = task_id
                    shard_state['shard_index'] = shard_idx
                    shard_state['skip_store'] = True
                    shard_states.append((
                        shard_state,
                        {
                            'configurable': {
                                'thread_id': shard_task_id
                            }
                        },
                    ))
                    remaining -= request_pairs
                    shard_idx += 1
                    logger.debug(
                        f'request_pairs: {request_pairs}, remaining: {remaining}, shard_idx: {shard_idx}'
                    )

                semaphore = asyncio.Semaphore(max(1, max_parallel_shards))

                async def run_shard(state_config: Tuple[Dict[str, Any],
                                                        Dict[str, Any]], ):
                    shard_state, shard_config = state_config
                    async with semaphore:
                        logger.info(
                            f'Launching shard {shard_state["task_id"]} '
                            f'for {shard_state["num_pairs"]} pairs')
                        return await self.workflow.ainvoke(
                            shard_state, shard_config)

                shard_tasks = [
                    asyncio.create_task(run_shard(sc)) for sc in shard_states
                ]
                shard_results = await asyncio.gather(*shard_tasks,
                                                     return_exceptions=True)

                successful_results: List[Dict[str, Any]] = []
                for idx, result in enumerate(shard_results):
                    if isinstance(result, Exception):
                        logger.error('Shard %d raised exception: %s', idx,
                                     result)
                        raise result
                    if not isinstance(result, dict):
                        logger.error(
                            f'Shard {idx} returned unexpected result type: {type(result)}'
                        )
                        return {
                            'task_id':
                            task_id,
                            'status':
                            TaskStatus.FAILED.value,
                            'error':
                            f'Unexpected shard result type: {type(result)}',
                        }
                    if result.get('status') == TaskStatus.FAILED.value:
                        logger.error(
                            f'Shard {idx} failed with errors: {result.get("error_messages")}'
                        )
                        return result
                    result['client'] = None
                    successful_results.append(result)

                if not successful_results:
                    logger.error('All shards failed for task %s', task_id)
                    return {
                        'task_id': task_id,
                        'status': TaskStatus.FAILED.value,
                        'error': 'All shards failed to produce output',
                    }

                # Merge shard outputs
                aggregate_state = deepcopy(successful_results[0])
                aggregate_state['task_id'] = task_id
                aggregate_state['parent_task_id'] = task_id
                aggregate_state['skip_store'] = False
                aggregate_state['shard_index'] = None
                aggregate_state['shard_results'] = [
                    res.get('task_id') for res in successful_results
                ]
                aggregate_state['enable_hard_negatives'] = initial_state.get(
                    'enable_hard_negatives', True)

                for key in (
                        'qa_pairs',
                        'validated_pairs',
                        'high_quality_qa_pairs',
                        'low_quality_qa_pairs',
                        'not_evaluated_pairs',
                ):
                    aggregate_state[key] = []

                def make_key(pair: Dict[str, Any]) -> Tuple[str, str]:
                    return (
                        pair.get('question', '').strip(),
                        pair.get('answer', '').strip(),
                    )

                merged_pairs: Dict[Tuple[str, str], Dict[str, Any]] = {}
                qa_order: List[Tuple[str, str]] = []
                validated_keys: List[Tuple[str, str]] = []
                high_keys: List[Tuple[str, str]] = []
                low_keys: List[Tuple[str, str]] = []
                not_eval_keys: List[Tuple[str, str]] = []

                for shard_result in successful_results:
                    for pair in shard_result.get('qa_pairs', []) or []:
                        key = make_key(pair)
                        if key not in merged_pairs:
                            merged_pairs[key] = deepcopy(pair)
                        if key not in qa_order:
                            qa_order.append(key)

                    for pair in shard_result.get('validated_pairs', []) or []:
                        key = make_key(pair)
                        if key not in merged_pairs:
                            merged_pairs[key] = deepcopy(pair)
                        if key not in validated_keys:
                            validated_keys.append(key)

                    for pair in (shard_result.get('high_quality_qa_pairs', [])
                                 or []):
                        key = make_key(pair)
                        if key not in merged_pairs:
                            merged_pairs[key] = deepcopy(pair)
                        if key not in high_keys:
                            high_keys.append(key)

                    for pair in (shard_result.get('low_quality_qa_pairs', [])
                                 or []):
                        key = make_key(pair)
                        if key not in merged_pairs:
                            merged_pairs[key] = deepcopy(pair)
                        if key not in low_keys:
                            low_keys.append(key)

                    for pair in (shard_result.get('not_evaluated_pairs', [])
                                 or []):
                        key = make_key(pair)
                        if key not in merged_pairs:
                            merged_pairs[key] = deepcopy(pair)
                        if key not in not_eval_keys:
                            not_eval_keys.append(key)

                ordered_keys: List[Tuple[str, str]] = []

                def append_keys(source: List[Tuple[str, str]]):
                    for key in source:
                        if key in merged_pairs and key not in ordered_keys:
                            ordered_keys.append(key)

                append_keys(high_keys)
                append_keys(not_eval_keys)
                append_keys(low_keys)
                append_keys(qa_order)

                if total_pairs and len(ordered_keys) > total_pairs:
                    ordered_keys = ordered_keys[:total_pairs]

                allowed_keys = set(ordered_keys)

                aggregate_state['qa_pairs'] = [
                    merged_pairs[key] for key in ordered_keys
                ]
                aggregate_state['validated_pairs'] = [
                    merged_pairs[key] for key in validated_keys
                    if key in allowed_keys
                ]
                aggregate_state['high_quality_qa_pairs'] = [
                    merged_pairs[key] for key in high_keys
                    if key in allowed_keys
                ]
                aggregate_state['low_quality_qa_pairs'] = [
                    merged_pairs[key] for key in low_keys
                    if key in allowed_keys
                ]
                aggregate_state['not_evaluated_pairs'] = [
                    merged_pairs[key] for key in not_eval_keys
                    if key in allowed_keys
                ]

                for bucket in (
                        'qa_pairs',
                        'validated_pairs',
                        'high_quality_qa_pairs',
                        'low_quality_qa_pairs',
                        'not_evaluated_pairs',
                ):
                    for pair in aggregate_state[bucket]:
                        pair['task_id'] = task_id

                enable_hard_negatives = aggregate_state.get(
                    'enable_hard_negatives', True)
                total_hard_negs = (sum(
                    len(pair.get('hard_negative_segment_ids', []))
                    for pair in aggregate_state['qa_pairs'])
                                   if enable_hard_negatives else 0)
                avg_hard_negs = (total_hard_negs /
                                 len(aggregate_state['qa_pairs'])
                                 if enable_hard_negatives
                                 and aggregate_state['qa_pairs'] else 0.0)
                aggregate_state['hard_negatives_stats'] = {
                    'total_hard_negatives': total_hard_negs,
                    'avg_per_question': avg_hard_negs,
                    'enabled': enable_hard_negatives,
                }

                # Update validation metrics
                total_validated = sum(
                    res.get('validation_results', {}).get(
                        'validated_pairs', 0) for res in successful_results)
                total_attempted = sum(
                    res.get('validation_results', {}).get('total_pairs', 0)
                    for res in successful_results)
                if total_attempted:
                    aggregate_state['validation_results'] = {
                        'total_pairs':
                        total_attempted,
                        'validated_pairs':
                        total_validated,
                        'dropped_pairs':
                        total_attempted - total_validated,
                        'validation_method':
                        successful_results[0].get('validation_results',
                                                  {}).get(
                                                      'validation_method',
                                                      'hybrid_with_faiss'),
                    }

                # Recompute evaluation and quality metrics based on combined lists
                evaluated_pairs = (aggregate_state['high_quality_qa_pairs'] +
                                   aggregate_state['low_quality_qa_pairs'])
                if evaluated_pairs:
                    avg_score = sum(
                        qa.get('evaluation', {}).get('overall', {}).get(
                            'score', 0)
                        for qa in evaluated_pairs) / len(evaluated_pairs)
                else:
                    avg_score = 0.0

                sample_eval = next(
                    (res.get('evaluation_metrics')
                     for res in successful_results
                     if res.get('evaluation_metrics')),
                    {},
                )

                aggregate_state['evaluation_metrics'] = {
                    'average_quality_score':
                    avg_score,
                    'evaluated_pairs':
                    len(evaluated_pairs),
                    'evaluation_model':
                    sample_eval.get('evaluation_model'),
                    'high_quality_ratio':
                    (len(aggregate_state['high_quality_qa_pairs']) /
                     max(len(evaluated_pairs), 1) if evaluated_pairs else 0.0),
                }

                quality_threshold = initial_state.get('evaluation', {}).get(
                    'quality_threshold', 6.0)
                aggregate_state['quality_metrics'] = {
                    'total_pairs':
                    len(aggregate_state['qa_pairs']),
                    'evaluated_pairs':
                    len(evaluated_pairs),
                    'high_quality_pairs':
                    len(aggregate_state['high_quality_qa_pairs']),
                    'low_quality_pairs':
                    len(aggregate_state['low_quality_qa_pairs']),
                    'not_evaluated_pairs':
                    len(aggregate_state['not_evaluated_pairs']),
                    'quality_threshold':
                    quality_threshold,
                    'average_quality':
                    avg_score,
                }

                aggregate_state['metrics'] = {
                    'task_id': task_id,
                    'qa_pairs_generated': len(aggregate_state['qa_pairs']),
                    'shards_completed': len(successful_results),
                    'pairs_per_shard': pairs_per_shard,
                    'total_requested_pairs': total_pairs,
                    'timestamp': datetime.now().isoformat(),
                }

                aggregate_state['qa_data_to_write'] = []
                for shard_result in successful_results:
                    qa_data_to_write = shard_result.get('qa_data_to_write', [])
                    logger.debug(
                        f'shard qa_data_to_write: {len(qa_data_to_write)}')
                    aggregate_state['qa_data_to_write'].extend(
                        qa_data_to_write)
                logger.info(
                    f'total shard qa_data_to_write: {len(aggregate_state["qa_data_to_write"])}'
                )

                aggregate_state['status'] = TaskStatus.COMPLETED.value

                # Persist combined results once
                store_response = await store_results(aggregate_state)
                aggregate_state.update(store_response or {})
                return aggregate_state

            # Fallback: single workflow run
            config = {'configurable': {'thread_id': task_id}}
            initial_state['client'] = self.client
            final_state = await self.workflow.ainvoke(initial_state, config)

            return final_state
        except Exception as e:
            error_msg = traceback.format_exc()
            logger.error('Error processing file %s: %s, %s', file_path, e,
                         error_msg)
            return {
                'task_id': task_id,
                'status': TaskStatus.FAILED.value,
                'error': str(e),
            }

    def write_all_qa_pairs(self, all_chunk_results, out_file):
        # Aggregate and write all QA pairs after processing all chunks

        def update_dict(parent_dict, key, val):
            if isinstance(val, dict):
                parent_dict.update(val)
            elif isinstance(val, list):
                parent_dict[key] = val

        try:
            combined_results = []
            total_qa_pairs = 0
            for result in all_chunk_results:
                if isinstance(result, dict) and result.get('qa_data_to_write'):
                    combined_result = {}
                    combined_result['qa_pairs'] = result.get(
                        'qa_data_to_write')
                    total_qa_pairs += len(combined_result['qa_pairs'])

                    top_artifacts = result.get('top_artifacts', {})
                    evaluation_metrics = result.get('evaluation_metrics', {})
                    quality_metrics = result.get('quality_metrics', {})
                    metrics = result.get('metrics', {})
                    segments = result.get('segments', [])

                    update_dict(combined_result, 'top_artifacts',
                                top_artifacts)
                    update_dict(
                        combined_result,
                        'evaluation_metrics',
                        evaluation_metrics,
                    )
                    update_dict(combined_result, 'quality_metrics',
                                quality_metrics)
                    update_dict(combined_result, 'metrics', metrics)
                    update_dict(combined_result, 'segments', segments)

                    combined_results.append(combined_result)

            logger.info(
                f'how many combined_results? {len(combined_results)}, total_qa_pairs: {total_qa_pairs}'
            )

            if combined_results:
                # Get distributions from config
                query_type_dist = self.config.get(
                    'query_type_distribution',
                    '{"multi_hop":0.4,"structural":0.3,"contextual":0.3}',
                )
                reasoning_type_dist = self.config.get(
                    'reasoning_type_distribution', None)

                question_only = self.config.get('question_only', False)
                logger.debug('question_only? %s', question_only)
                if not question_only:
                    logger.debug(
                        f'combined results[0]: {combined_results[0].keys()}')
                    all_qa_pairs = []
                    for re in combined_results:
                        all_qa_pairs.extend(re['qa_pairs'])
                        logger.info('num all_qa_pairs: %d', len(all_qa_pairs))
                    metrics = compute_metrics(all_qa_pairs, query_type_dist,
                                              reasoning_type_dist)
                    logger.info(
                        f'Final aggregated metrics across all chunks: {metrics}'
                    )
                else:
                    logger.info('Skip compute_metrics')

                # append to jsonl
                with open(out_file, 'a') as fout:
                    for elem in combined_results:
                        line = json.dumps(convert_posixpath_to_string(elem))
                        fout.write(line + '\n')
                    # json.dump(all_qa_pairs, fout, indent=2)
                    logger.info(
                        f'Wrote {len(combined_results)} QA pairs to {out_file}'
                    )

            else:
                logger.warning('No QA pairs generated across all chunks')

        except Exception as e:
            error_msg = traceback.format_exc()
            logger.error('Error write_all_qa_pairs: %s, %s', e, error_msg)

    def visualize_workflow(self):
        """Generate a visual representation of the workflow"""
        try:
            # This would generate a Mermaid diagram or similar
            return self.workflow.get_graph().draw_mermaid()
        except Exception as e:
            logger.error(e)
            return "Visualization not available"


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='TrueQuery QA Generation System')
    parser.add_argument(
        '--config',
        '-c',
        type=str,
        required=True,
        default='txt2qa_config/text_config.yaml',
        help='Path to configuration YAML file (required).',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help=
        'Overwrite existing output files and reprocess all input files (disables resume functionality).',
    )

    parser.add_argument(
        '--batch',
        type=int,
        default=0,
        help=
        'Which batch to run. For example, if batch_size = 10, we will process input files from 0 to 9 with batch 0, and 10 to 19 with batch 1 etc',
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='Number of input files to process in each batch',
    )

    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help=
        'Enable verbose logging (INFO and DEBUG messages). By default, only WARNING/ERROR/CRITICAL are shown.',
    )

    args = parser.parse_args()

    # Configure logging based on verbose flag
    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        )

    # Load configuration from file
    config_path = args.config
    if not os.path.exists(config_path):
        logger.error('Configuration file not found: %s', config_path)
        exit(1)

    logger.info('Loading configuration from: %s', config_path)

    # Run the system
    generator = QAGenerator(
        config_path,
        overwrite=args.overwrite,
        batch=args.batch,
        batch_size=args.batch_size,
    )

    # Process files
    asyncio.run(generator.process_directory())
    # logger.info(generator.visualize_workflow())
