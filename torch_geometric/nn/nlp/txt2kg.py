from typing import List, Optional, Tuple
import math

class TXT2KG():
    """Uses NVIDIA NIMs + Prompt engineering to extract KG from text
    nvidia/llama-3.1-nemotron-70b-instruct is on par or better than GPT4o
    in benchmarks. We need a high quality model to ensure high quality KG.
    Otherwise garbage in garbage out.
    """
    def __init__(
        self,
        NVIDIA_API_KEY,
        chunk_size=512,
    ) -> None:
        # We use NIMs since most PyG users may not be able to run a 70B+ model
        from openai import OpenAI

        self.client = OpenAI(base_url="https://integrate.api.nvidia.com/v1",
                             api_key=NVIDIA_API_KEY)
        self.chunk_size = 512
        self.system_prompt = "Please convert the above text into a list of knowledge triples with the form ('entity', 'relation', 'entity'). Seperate each with a new line. Do not output anything else.”"
        self.model = "nvidia/llama-3.1-nemotron-70b-instruct"
        self.triples_per_doc_id = {}
        # keep track of which doc each triple comes from
        # useful for approximating recall of subgraph retrieval algos
        self.doc_id_counter = 0
        self.relevant_docs_per_q_a_pair = {}

    def chunk_to_triples_str(self, txt: str) -> str:
        # call LLM on text
        completion = self.client.chat.completions.create(
            model=self.model, messages=[{
                "role":
                "user",
                "content":
                txt + '\n' + self.system_prompt
            }], temperature=0, top_p=1, max_tokens=1024, stream=True)
        out_str = ""
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                out_str += chunk.choices[0].delta.content
        return out_str

    def parse_n_check_triples(self,
                              triples_str: str) -> List[Tuple[str, str, str]]:
        # use pythonic checks for triples
        print(triples_str)
        # (TODO) make pythonic logic to parse into triples

    def add_doc_2_KG(
        self,
        txt: str,
        QA_pair: Optional[Tuple[str, str]],
    ) -> None:
        # if QA_pair is not None, store with matching doc ids
        # useful for approximating recall
        chunks = [
            txt[i:min((i + 1) * self.chunk_size, len(txt))]
            for i in range(math.ceil(len(txt) / self.chunk_size))
        ]
        self.triples_per_doc_id[self.doc_id_counter] = []
        for chunk in chunks:
            self.triples_per_doc_id[
                self.doc_id_counter] += self.parse_n_check_triples(
                    self.chunk_to_triples_str(chunk))
        if QA_pair:
            if QA_pair in self.relevant_docs_per_q_a_pair.keys():
                self.relevant_docs_per_q_a_pair[QA_pair] += [
                    self.doc_id_counter
                ]
            else:
                self.relevant_docs_per_q_a_pair[QA_pair] = [
                    self.doc_id_counter
                ]
        self.doc_id_counter += 1

kg_maker = TXT2KG(
    NVIDIA_API_KEY="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",
    chunk_size=512,
)
kg_maker.add_doc_2_KG(
    txt='Radio City is India\'s first private FM radio station and was started on 3 July 2001. It broadcasts on 91.1 (earlier 91.0 in most cities) megahertz from Mumbai (where it was started in 2004), Bengaluru (started first in 2001), Lucknow and New Delhi (since 2003). It plays Hindi, English and regional songs. It was launched in Hyderabad in March 2006, in Chennai on 7 July 2006 and in Visakhapatnam October 2007. Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features. The Radio station currently plays a mix of Hindi and Regional music. Abraham Thomas is the CEO of the company.Football in Albania existed before the Albanian Football Federation (FSHF) was created. This was evidenced by the team\'s registration at the Balkan Cup tournament during 1929-1931, which started in 1929 (although Albania eventually had pressure from the teams because of competition, competition started first and was strong enough in the duels) . Albanian National Team was founded on June 6, 1930, but Albania had to wait 16 years to play its first international match and then defeated Yugoslavia in 1946. In 1932, Albania joined FIFA (during the 12–16 June convention ) And in 1954 she was one of the founding members of UEFA.Echosmith is an American, Corporate indie pop band formed in February 2009 in Chino, California. Originally formed as a quartet of siblings, the band currently consists of Sydney, Noah and Graham Sierota, following the departure of eldest sibling Jamie in late 2016. Echosmith started first as "Ready Set Go!" until they signed to Warner Bros. Records in May 2012. They are best known for their hit song "Cool Kids", which reached number 13 on the "Billboard" Hot 100 and was certified double platinum by the RIAA with over 1,200,000 sales in the United States and also double platinum by ARIA in Australia. The song was Warner Bros. Records\' fifth-biggest-selling-digital song of 2014, with 1.3 million downloads sold. The band\'s debut album, "Talking Dreams", was released on October 8, 2013.Women\'s colleges in the Southern United States refers to undergraduate, bachelor\'s degree–granting institutions, often liberal arts colleges, whose student populations consist exclusively or almost exclusively of women, located in the Southern United States. Many started first as girls\' seminaries or academies. Salem College is the oldest female educational institution in the South and Wesleyan College is the first that was established specifically as a college for women. Some schools, such as Mary Baldwin University and Salem College, offer coeducational courses at the graduate level.The First Arthur County Courthouse and Jail, was perhaps the smallest court house in the United States, and serves now as a museum.Arthur\'s Magazine (1844–1846) was an American literary periodical published in Philadelphia in the 19th century. Edited by T.S. Arthur, it featured work by Edgar A. Poe, J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others. In May 1846 it was merged into "Godey\'s Lady\'s Book".The 2014–15 Ukrainian Hockey Championship was the 23rd season of the Ukrainian Hockey Championship. Only four teams participated in the league this season, because of the instability in Ukraine and that most of the clubs had economical issues. Generals Kiev was the only team that participated in the league the previous season, and the season started first after the year-end of 2014. The regular season included just 12 rounds, where all the teams went to the semifinals. In the final, ATEK Kiev defeated the regular season winner HK Kremenchuk.First for Women is a woman\'s magazine published by Bauer Media Group in the USA. The magazine was started in 1989. It is based in Englewood Cliffs, New Jersey. In 2011 the circulation of the magazine was 1,310,696 copies.The Freeway Complex Fire was a 2008 wildfire in the Santa Ana Canyon area of Orange County, California. The fire started as two separate fires on November 15, 2008. The "Freeway Fire" started first shortly after 9am with the "Landfill Fire" igniting approximately 2 hours later. These two separate fires merged a day later and ultimately destroyed 314 residences in Anaheim Hills and Yorba Linda.William Rast is an American clothing line founded by Justin Timberlake and Trace Ayala. It is most known for their premium jeans. On October 17, 2006, Justin Timberlake and Trace Ayala put on their first fashion show to launch their new William Rast clothing line. The label also produces other clothing items such as jackets and tops. The company started first as a denim line, later evolving into a men’s and women’s clothing line.',
    QA_pair=("Which magazine was started first Arthur's Magazine or First for Women?", "Arthur's Magazine"),
)
