Working with LLM RAG in Pytorch Geometric
=========================================

This series aims to provide a starting point and for 
multi-step LLM Retrieval Augmented Generation
(RAG) using Graph Neural Networks.

Motivation
----------

As Large Language Models (LLMs) quickly grow to dominate industry, they
are increasingly being deployed at scale in use cases that require very
specific contextual expertise. LLMs often struggle with these cases out
of the box, as they will hallucinate answers that are not included in
their training data. At the same time, many business already have large
graph databases full of important context that can provide important
domain-specific context to reduce hallucination and improve answer
fidelity for LLMs. Graph Neural Networks (GNNs) provide a means for
efficiently encoding this contextual information into the model, which
can help LLMs to better understand and generate answers. Hence, theres
an open research question as to how to effectively use GNN encodings
efficiently for this purpose, that the tooling provided here can help
investigate.

Architecture
------------

To model the use-case of RAG from a large knowledge graph of millions of
nodes, we present the following architecture:





.. image:: _Introduction_files/_Introduction_7_0.svg



Graph RAG as shown in the diagram above follows the following order of
operations:

0. To start, not pictured here, there must exist a large knowledge graph
   that exists as a source of truth. The nodes and edges of this
   knowledge graph

During inference time, RAG implementations that follow this architecture
are composed of the following steps:

1. Tokenize and encode the query using the LLM Encoder
2. Retrieve a subgraph of the larger knowledge graph (KG) relevant to
   the query and encode it using a GNN
3. Jointly embed the GNN embedding with the LLM embedding
4. Utilize LLM Decoder to decode joint embedding and generate a response




Encoding a Large Knowledge Graph
================================

In this notebook, we are going to walk through how to encode a large
knowledge graph for the purposes of Graph RAG. We will provide two
examples of how to do so, along with demonstration code.

Example 1: Building from Already Existing Datasets
--------------------------------------------------

In most RAG scenarios, the subset of the information corpus that gets
retrieved is crucial for whether the appropriate response to the LLM.
The same is true for GNN based RAG. Consider the following dataset
WebQSP:

.. code:: ipython3

    from torch_geometric.datasets import UpdatedWebQSPDataset
    
    num_questions = 100
    ds = UpdatedWebQSPDataset('small_sample', limit=num_questions)


.. parsed-literal::

    /home/zaristei/.local/lib/python3.10/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm


WebQSP is a dataset that is based off of a subset of the Freebase
Knowledge Graph, which is an open-source knowledge graph formerly
maintained by Google. For each question-answer pair in the dataset, a
subgraph was chosen based on a Semantic SPARQL search on the larger
knowledge graph, to provide relevent context on finding the answer. So
each entry in the dataset consists of: - A question to be answered - The
answer - A knowledge graph subgraph of Freebase that has the context
needed to answer the question.

.. code:: ipython3

    ds.raw_dataset




.. parsed-literal::

    Dataset({
        features: ['id', 'question', 'answer', 'q_entity', 'a_entity', 'graph', 'choices'],
        num_rows: 100
    })



.. code:: ipython3

    ds.raw_dataset[0]




.. parsed-literal::

    {'id': 'WebQTrn-0',
     'question': 'what is the name of justin bieber brother',
     'answer': ['Jaxon Bieber'],
     'q_entity': ['Justin Bieber'],
     'a_entity': ['Jaxon Bieber'],
     'graph': [['P!nk', 'freebase.valuenotation.is_reviewed', 'Gender'],
      ['1Club.FM: Power', 'broadcast.content.artist', 'P!nk'],
      ['Somebody to Love', 'music.recording.contributions', 'm.0rqp4h0'],
      ['Rudolph Valentino',
       'freebase.valuenotation.is_reviewed',
       'Place of birth'],
      ['Ice Cube', 'broadcast.artist.content', '.977 The Hits Channel'],
      ['Colbie Caillat', 'broadcast.artist.content', 'Hot Wired Radio'],
      ['Stephen Melton', 'people.person.nationality', 'United States of America'],
      ['Record producer',
       'music.performance_role.regular_performances',
       'm.012m1vf1'],
      ['Justin Bieber', 'award.award_winner.awards_won', 'm.0yrkc0l'],
      ['1.FM Top 40', 'broadcast.content.artist', 'Geri Halliwell'],
      ['2011 Teen Choice Awards',
       'award.award_ceremony.awards_presented',
       'm.0yrkr34'],
      ['m.012bm2v1', 'celebrities.friendship.friend', 'Miley Cyrus'],
      ['As Long As You Love Me (Ferry Corsten radio)',
       'common.topic.notable_types',
       'Musical Recording'],
      ['Toby Gad', 'music.artist.genre', 'Rhythm and blues'],
      ['Stratford', 'location.location.containedby', 'Canada'],
      ['Singer',
       'base.lightweight.profession.specialization_of',
       'Musicians and Singers'],
      ['Enrique Iglesias', 'people.person.profession', 'Singer'],
      ['Beauty and a Beat (acoustic version)',
       'music.recording.artist',
       'Justin Bieber'],
      ['Akon', 'freebase.valuenotation.is_reviewed', 'Place of birth'],
      ['.977 The Hits Channel', 'broadcast.content.artist', 'Britney Spears'],
      ['50 Cent', 'people.person.profession', 'Film Producer'],
      ['As Long As You Love Me (Audien dubstep mix)',
       'music.recording.canonical_version',
       'As Long As You Love Me'],
      ['Kevin Risto', 'people.person.gender', 'Male'],
      ['Classic Soul Network', 'common.topic.notable_types', 'Broadcast Content'],
      ['Shaggy', 'broadcast.artist.content', 'HitzRadio.com'],
      ['Mary J. Blige', 'people.person.profession', 'Record producer'],
      ['Live My Life', 'common.topic.notable_for', 'g.12ml2glpn'],
      ['Paul Anka', 'common.topic.notable_types', 'Musical Artist'],
      ['m.0_w1gn3', 'award.award_nomination.nominated_for', 'Change Me'],
      ['Baby', 'award.award_winning_work.awards_won', 'm.0n1ykxp'],
      ['m.0njhxd_', 'award.award_honor.award_winner', 'Justin Bieber'],
      ['1Club.FM: V101', 'broadcast.content.artist', 'The Roots'],
      ['#thatPOWER', 'music.recording.tracks', '#thatPOWER'],
      ['m.0ghz3d6', 'tv.tv_guest_role.actor', 'Justin Bieber'],
      ['American Music Award for Favorite Pop/Rock Album',
       'award.award_category.winners',
       'm.0ndc259'],
      ['A Michael Bublé Christmas', 'film.film.personal_appearances', 'm.0ng_vkd'],
      ['Ontario', 'location.administrative_division.country', 'Canada'],
      ['1Club.FM: Power', 'common.topic.notable_types', 'Broadcast Content'],
      ['Music Producer', 'common.topic.subject_of', 'POPPMusic.net'],
      ['Billboard Music Award for Top Streaming Artist',
       'award.award_category.winners',
       'm.0njhx1b'],
      ['Justin Bieber', 'film.producer.film', "Justin Bieber's Believe"],
      ['Heartbreaker', 'music.composition.recordings', 'Heartbreaker'],
      ['Brandy Norwood', 'people.person.profession', 'Singer'],
      ["Justin Bieber's Believe", 'film.film.personal_appearances', 'm.0101ft2j'],
      ['Justin Bieber', 'music.artist.album', 'All Bad'],
      ['m.0n4rmg7', 'freebase.valuenotation.is_reviewed', 'Ceremony'],
      ['m.0v_729v',
       'tv.tv_guest_personal_appearance.episode',
       'Results Show: Week 7'],
      ['Big R Radio - Top 40 Hits', 'broadcast.content.artist', 'Britney Spears'],
      ['One Less Lonely Girl',
       'music.album.primary_release',
       'One Less Lonely Girl'],
      ['Twista', 'people.person.gender', 'Male'],
      ['1Club.FM: Channel One', 'broadcast.content.artist', 'Eminem'],
      ['Ciara', 'broadcast.artist.content', 'FLOW 103'],
      ['Jon M. Chu', 'film.director.film', "Justin Bieber's Believe"],
      ['Leonardo DiCaprio', 'freebase.valuenotation.is_reviewed', 'Date of birth'],
      ['m.0ndc3_1', 'freebase.valuenotation.has_no_value', 'Winning work'],
      ['Somebody To Love', 'music.recording.artist', 'Justin Bieber'],
      ['Toby Gad', 'music.artist.genre', 'Rock music'],
      ['Madonna', 'music.artist.genre', 'Pop music'],
      ['Selena Gomez', 'music.artist.genre', 'Europop'],
      ['m.0gbm3cg',
       'film.personal_film_appearance.film',
       'Justin Bieber: Never Say Never'],
      ['Baby', 'music.recording.canonical_version', 'Baby'],
      ['Contemporary R&B', 'music.genre.subgenre', 'Quiet Storm'],
      ['Boyfriend', 'music.recording.artist', 'Justin Bieber'],
      ['Dr. Dre', 'music.artist.genre', 'Rap music'],
      ['MTV Video Music Award Japan for Best New Artist',
       'award.award_category.winners',
       'm.0yrhrwc'],
      ['Beauty and a Beat', 'music.recording.featured_artists', 'Nicki Minaj'],
      ['Hip hop music', 'broadcast.genre.content', 'FLOW 103'],
      ['Maroon 5', 'broadcast.artist.content', '1Club.FM: Mix 106'],
      ['m.0gctwjk',
       'tv.tv_guest_role.episodes_appeared_in',
       'Series 2, Episode 3'],
      ['Enrique Iglesias', 'music.artist.genre', 'Dance-pop'],
      ['Beauty and a Beast', 'music.recording.artist', 'Justin Bieber'],
      ['FLOW 103', 'broadcast.content.genre', 'Hip hop music'],
      ['Madonna', 'broadcast.artist.content', 'radioIO RNB Mix'],
      ['Selena Gomez', 'people.person.profession', 'Dancer'],
      ['Little Bird', 'music.recording.tracks', 'm.0v2hrym'],
      ['Juno Fan Choice Award', 'award.award_category.winners', 'm.0t4s_bn'],
      ['Never Say Never', 'common.topic.notable_types', 'Musical Recording'],
      ['As Long As You Love Me (PAULO & JACKINSKY radio)',
       'common.topic.notable_types',
       'Musical Recording'],
      ['Beauty and a Beat',
       'music.single.versions',
       'Beauty and a Beat (Wideboys Club Mix)'],
      ['Carrie Underwood', 'freebase.valuenotation.is_reviewed', 'Date of birth'],
      ['Bryan Adams',
       'freebase.valuenotation.is_reviewed',
       'Country of nationality'],
      ['Madonna', 'people.person.profession', 'Singer-songwriter'],
      ['Gavin DeGraw', 'broadcast.artist.content', '1Club.FM: Mix 106'],
      ['Iggy Azalea', 'freebase.valuenotation.is_reviewed', 'Gender'],
      ['m.0ndc259', 'award.award_honor.award_winner', 'Justin Bieber'],
      ['Terence Dudley', 'music.artist.genre', 'Reggae'],
      ['Kylie Minogue', 'people.person.profession', 'Actor'],
      ['Adrienne Bailon', 'music.artist.genre', 'Pop music'],
      ['Katy Perry', 'music.artist.genre', 'Electronic music'],
      ['Dany Brillant', 'people.person.gender', 'Male'],
      ['Martin Kierszenbaum', 'people.person.gender', 'Male'],
      ['Anastacia', 'people.person.nationality', 'United States of America'],
      ['Amerie', 'music.artist.label', 'The Island Def Jam Music Group'],
      ['Madonna', 'freebase.valuenotation.is_reviewed', 'Children'],
      ['Gwen Stefani', 'freebase.valuenotation.is_reviewed', 'Date of birth'],
      ['Somebody to Love', 'music.composition.form', 'Song'],
      ['Teen Choice Award for Choice Twitter Personality',
       'award.award_category.winners',
       'm.0yrkr34'],
      ['Chef Tone', 'people.person.place_of_birth', 'Chicago'],
      ['Dan Cutforth', 'freebase.valuenotation.has_value', 'Parents'],
      ['Selena Gomez', 'freebase.valuenotation.is_reviewed', 'Parents'],
      ['SoulfulHipHop.com Radio', 'broadcast.content.artist', 'Whitney Houston'],
      ['Record producer',
       'fictional_universe.character_occupation.characters_with_this_occupation',
       'Haley James Scott'],
      ['Colbie Caillat', 'music.artist.genre', 'Pop music'],
      ['C1', 'music.artist.genre', 'Contemporary R&B'],
      ['Pattie Mallette', 'people.person.spouse_s', 'm.0101gx29'],
      ['Emphatic Radio.com!', 'broadcast.content.artist', 'Kid Cudi'],
      ['Kanye West', 'people.person.profession', 'Singer'],
      ['Pop music', 'common.topic.subject_of', 'Stephen Melton'],
      ['radioIO Todays POP', 'broadcast.content.producer', 'Radioio'],
      ['Emphatic Radio.com!', 'broadcast.content.artist', 'Shaffer Smith'],
      ['Avril Lavigne', 'broadcast.artist.content', '1Club.FM: Channel One'],
      ['m.03vbp89', 'common.image.appears_in_topic_gallery', 'HitzRadio.com'],
      ['Mannie Fresh', 'freebase.valuenotation.has_value', 'Height'],
      ['Big R Radio - Top 40 Hits', 'broadcast.content.artist', 'Flyleaf'],
      ['Jennifer Lopez', 'music.artist.genre', 'Contemporary R&B'],
      ['Beauty And A Beat', 'music.composition.recordings', 'Beauty And A Beat'],
      ['Rihanna', 'broadcast.artist.content', 'WildFMRadio.com'],
      ['Adam Messinger', 'music.composer.compositions', 'Mistletoe'],
      ['Live My Life', 'music.album.compositions', 'Live My Life'],
      ['RedOne', 'music.artist.genre', 'Rock music'],
      ['#thatPOWER', 'music.recording.canonical_version', '#thatPOWER'],
      ['m.0yrjkl1', 'award.award_honor.honored_for', 'Baby'],
      ['Terius Nash', 'music.artist.genre', 'Rhythm and blues'],
      ['Little Bird', 'common.topic.notable_types', 'Musical Recording'],
      ['As Long As You Love Me (Ferry Corsten radio)',
       'music.recording.featured_artists',
       'Big Sean'],
      ['Mary J. Blige', 'broadcast.artist.content', 'HitzRadio.com'],
      ['m.0gxnp5d', 'base.popstra.hangout.customer', 'Justin Bieber'],
      ['Terius Nash', 'people.person.nationality', 'United States of America'],
      ['Justin Bieber', 'tv.tv_program_guest.appeared_on', 'm.0_grmr_'],
      ['Athan Grace', 'people.person.profession', 'Actor'],
      ['SoulfulHipHop.com Radio', 'broadcast.content.genre', 'Hip hop music'],
      ['Shorty Award for Music', 'award.award_category.nominees', 'm.0z3tqqt'],
      ['All Around the World (acoustic version)',
       'music.recording.artist',
       'Justin Bieber'],
      ['Bad Day', 'music.composition.composer', 'Marvin Isley'],
      ['Brandy Norwood',
       'influence.influence_node.influenced_by',
       'Whitney Houston'],
      ['Duffy', 'freebase.valuenotation.is_reviewed', 'Date of birth'],
      ['MTV Video Music Award for Artist to Watch',
       'award.award_category.winners',
       'm.0n1ykxp'],
      ['Caitlin Beadles',
       'celebrities.celebrity.sexual_relationships',
       'm.0d33gyj'],
      ['As Long As You Love Me',
       'music.single.versions',
       'As Long As You Love Me (Audiobot instrumental)'],
      ['Emphatic Radio.com!', 'common.topic.image', 'Emphatic Radio.com!'],
      ['Singer', 'base.descriptive_names.names.descriptive_name', 'm.0111fg4h'],
      ['School Boy Records', 'music.record_label.artist', 'Scooter Braun'],
      ['Lupe Fiasco', 'freebase.valuenotation.is_reviewed', 'Date of birth'],
      ['Zac Efron', 'base.icons.icon.icon_genre', 'Teen idol'],
      ['The Island Def Jam Music Group',
       'music.record_label.artist',
       'The Mighty Mighty Bosstones'],
      ['m.012bm3j9', 'celebrities.friendship.friend', 'Rita Ora'],
      ['Toby Gad', 'music.lyricist.lyrics_written', 'Beautiful'],
      ['Lolly', 'music.composition.composer', 'Juicy J'],
      ['Justin Bieber: Never Say Never',
       'media_common.netflix_title.netflix_genres',
       'Documentary film'],
      ['Timbaland', 'freebase.valuenotation.is_reviewed', 'Date of birth'],
      ['m.0z1scxk', 'freebase.valuenotation.has_no_value', 'Winning work'],
      ['Love Me', 'common.topic.notable_for', 'g.12h2xd7m9'],
      ['Trey Songz', 'freebase.valuenotation.is_reviewed', 'Place of birth'],
      ['Amerie', 'music.artist.genre', 'Pop music'],
      ['1.FM Top 40', 'broadcast.content.artist', 'Beyoncé Knowles'],
      ['The Island Def Jam Music Group', 'music.record_label.artist', 'Y?N-Vee'],
      ['Rodney Jerkins', 'music.artist.genre', 'Synthpop'],
      ['WildFMRadio.com', 'broadcast.content.artist', 'Soulja Boy'],
      ['As Long As You Love Me',
       'music.single.versions',
       'As Long As You Love Me (Audien dubstep edit)'],
      ['Will Smith', 'broadcast.artist.content', 'Sunshine Radio'],
      ['Recovery', 'music.recording.song', 'Recovery'],
      ['Justin Timberlake', 'music.artist.genre', 'Electronic music'],
      ['Mannie Fresh', 'people.person.nationality', 'United States of America'],
      ['m.0101ftqp', 'film.film_crew_gig.film', "Justin Bieber's Believe"],
      ['Benny Blanco', 'common.topic.notable_types', 'Record Producer'],
      ['Leif Garrett', 'music.artist.genre', 'Rock music'],
      ['Annette Funicello', 'freebase.valuenotation.is_reviewed', 'Date of birth'],
      ['WildFMRadio.com', 'broadcast.content.artist', 'The Black Eyed Peas'],
      ['First Dance', 'music.recording.artist', 'Justin Bieber'],
      ['#thatPower', 'music.recording.song', '#thatPower'],
      ['Children', 'rdf-schema#range', 'Person'],
      ['Beautiful', 'common.topic.notable_for', 'g.1256glpl9'],
      ['Kid Cudi', 'broadcast.artist.content', 'Emphatic Radio.com!'],
      ['Lady Gaga', 'broadcast.artist.content', 'Emphatic Radio.com!'],
      ['2013 Teen Choice Awards',
       'award.award_ceremony.awards_presented',
       'm.0wjgqck'],
      ['The Island Def Jam Music Group',
       'organization.organization.parent',
       'm.04q65lb'],
      ['The Island Def Jam Music Group',
       'music.record_label.artist',
       'Rusted Root'],
      ['radioIO RNB Mix', 'common.topic.notable_types', 'Broadcast Content'],
      ['m.0z87d3n',
       'award.award_honor.award',
       'Teen Choice Award for Choice Red Carpet Fashion Icon Male'],
      ['Shaffer Smith', 'music.artist.genre', 'Dance music'],
      ['Live My Life', 'music.composition.composer', 'John Mamann'],
      ['radioIO Classic RNB', 'broadcast.content.genre', 'Rock music'],
      ['m.0njw4z2', 'award.award_honor.award_winner', 'Justin Bieber'],
      ['P!nk', 'freebase.valuenotation.is_reviewed', 'Profession'],
      ['Ludacris', 'broadcast.artist.content', 'SoulfulHipHop.com Radio'],
      ['Trick Daddy', 'broadcast.artist.content', 'PowerHitz'],
      ['1.FM Top 40', 'broadcast.content.artist', 'Yellowcard'],
      ['Tampa', 'location.location.containedby', 'United States of America'],
      ['Love Never Felt So Good',
       'music.album.compositions',
       'Love Never Felt So Good'],
      ['As Long As You Love Me (Ferry Corsten remix)',
       'music.recording.artist',
       'Justin Bieber'],
      ['Nelly', 'music.artist.genre', 'Rhythm and blues'],
      ['Marvin Isley', 'music.composer.compositions', 'Bad Day'],
      ['Somebody to Love', 'common.topic.notable_types', 'Composition'],
      ['Katy Perry', 'broadcast.artist.content', '1Club.FM: Power'],
      ['Snoop Dogg', 'people.person.gender', 'Male'],
      ['DMX', 'broadcast.artist.content', '.977 The Hits Channel'],
      ['Singer', 'base.descriptive_names.names.descriptive_name', 'm.0109_45q'],
      ['Estelle', 'people.person.profession', 'Record producer'],
      ['m.0_syttc', 'award.award_nomination.award_nominee', 'Justin Bieber'],
      ['PowerHitz', 'broadcast.content.genre', 'Hip hop music'],
      ['Chris Brown', 'broadcast.artist.content', 'Big R Radio - The Hawk'],
      ['50 Cent', 'people.person.nationality', 'United States of America'],
      ['Chris Jasper', 'people.person.gender', 'Male'],
      ['Sir Nolan', 'music.artist.genre', 'Pop music'],
      ['Hot Wired Radio', 'broadcast.content.producer', 'Hot Wired Radio'],
      ['m.0v_6zk4', 'tv.tv_guest_personal_appearance.person', 'Justin Bieber'],
      ['Snoop Dogg',
       'freebase.valuenotation.is_reviewed',
       'Country of nationality'],
      ['David Nicksay', 'people.person.gender', 'Male'],
      ['Justin Bieber', 'people.person.profession', 'Record producer'],
      ['Everlast', 'people.person.profession', 'Singer-songwriter'],
      ['Juno Awards of 2014',
       'award.award_ceremony.awards_presented',
       'm.0102z0vx'],
      ['As Long As You Love Me (Audiobot remix)',
       'music.recording.song',
       'As Long as You Love Me'],
      ['#thatPower', 'music.composition.composer', 'Will i Am'],
      ['m.0gbm3bl', 'film.personal_film_appearance.person', 'Miley Cyrus'],
      ['m.0_cyzs_',
       'celebrities.legal_entanglement.offense',
       'Driving under the influence'],
      ['LeAnn Rimes', 'people.person.profession', 'Actor'],
      ['KooL CrAzE', 'music.artist.label', 'The Island Def Jam Music Group'],
      ['1Club.FM: Power', 'broadcast.content.artist', 'Usher'],
      ['Mann', 'people.person.gender', 'Male'],
      ['JoJo', 'people.person.gender', 'Female'],
      ['Right Here (featuring Drake)',
       'music.recording.canonical_version',
       'Right Here'],
      ['Mason Levy', 'music.composer.compositions', 'Boyfriend'],
      ['Beauty and a Beat', 'music.recording.artist', 'Justin Bieber'],
      ['m.0yrjynf',
       'award.award_honor.award',
       'Teen Choice Award for Choice Summer Music Star: Male'],
      ['Pras', 'people.person.profession', 'Record producer'],
      ['1Club.FM: Power', 'broadcast.content.artist', 'Daniel Bedingfield'],
      ['Hold Tight', 'award.award_nominated_work.award_nominations', 'm.0_w3zrs'],
      ['My World 2.0', 'music.album.releases', 'My World 2.0'],
      ['Mannie Fresh', 'freebase.valuenotation.is_reviewed', 'Date of birth'],
      ['Christmas in Washington', 'film.film.personal_appearances', 'm.0ng_k21'],
      ['Marvin Isley',
       'freebase.valuenotation.is_reviewed',
       'Country of nationality'],
      ['Raekwon', 'broadcast.artist.content', 'Smoothbeats'],
      ['Adam Messinger', 'freebase.valuenotation.has_value', 'Parents'],
      ['Adam Messinger', 'freebase.valuenotation.has_value', 'Date of birth'],
      ['My World 2.0', 'common.topic.webpage', 'm.0cvc8k4'],
      ['Justin Bieber', 'tv.tv_actor.guest_roles', 'm.0gctytd'],
      ['Emphatic Radio.com!', 'broadcast.content.artist', 'Linkin Park'],
      ['Toby Gad', 'freebase.valuenotation.is_reviewed', 'Date of birth'],
      ['School Gyrls', 'film.film.language', 'English Language'],
      ['Jordin Sparks', 'music.artist.genre', 'Contemporary R&B'],
      ['Boyfriend', 'music.composition.recordings', 'Boys / Boyfriend'],
      ['Katy Perry', 'people.person.profession', 'Actor'],
      ['As Long as You Love Me', 'common.topic.notable_for', 'g.125ddwtp0'],
      ['Ronald Isley', 'people.person.profession', 'Actor'],
      ['Live My Life (Party Rock remix)',
       'music.recording.featured_artists',
       'Redfoo'],
      ['HitzRadio.com', 'common.topic.webpage', 'm.03zb5cw'],
      ['Jaxon Bieber', 'people.person.nationality', 'Canada'],
      ['As Long as You Love Me (album version)',
       'common.topic.notable_types',
       'Musical Recording'],
      ['Justin Bieber: Just Getting Started',
       'book.written_work.author',
       'Justin Bieber'],
      ['BeirutNights.com Radio',
       'broadcast.content.artist',
       'Marc Maris vs. Ramone'],
      ['Gwen Stefani', 'people.person.profession', 'Musician'],
      ['m.0pcnqnb', 'film.personal_film_appearance.person', 'Justin Bieber'],
      ['m.0101fsyr', 'film.personal_film_appearance.person', 'Scooter Braun'],
      ['Singer', 'base.descriptive_names.names.descriptive_name', 'm.0102gvnb'],
      ['Justin Bieber', 'music.featured_artist.recordings', '#Thatpower'],
      ['Justin Bieber', 'celebrities.celebrity.net_worth', 'm.0yqflrk'],
      ['Love Never Felt So Good',
       'music.album.releases',
       'Love Never Felt So Good'],
      ['Hot Wired Radio', 'broadcast.content.artist', 'Shaffer Smith'],
      ['BeirutNights.com Radio', 'broadcast.content.artist', 'Soundlovers'],
      ['Beauty and a Beat (DJ Laszlo Body Rock Radio Mix)',
       'music.recording.canonical_version',
       'Beauty and a Beat'],
      ['Sir Mix-a-Lot', 'people.person.profession', 'Actor'],
      ['Big R Radio - Top 40 Hits', 'broadcast.content.artist', 'Usher'],
      ['Dance music',
       'broadcast.genre.content',
       "PartyRadioUSA.net - Nation's #1 Party Authority"],
      ['1Club.FM: V101', 'broadcast.content.location', 'Chicago'],
      ['Terius Nash', 'people.person.profession', 'Record producer'],
      ['Terence Dudley', 'people.person.profession', 'Record producer'],
      ['Mary J. Blige', 'common.topic.notable_types', 'Musical Artist'],
      ['Baby', 'common.topic.notable_types', 'Award-Winning Work'],
      ['Lolly', 'music.recording.canonical_version', 'Lolly'],
      ['Scooter Braun', 'people.person.gender', 'Male'],
      ['Mistletoe', 'music.album.artist', 'Justin Bieber'],
      ['Sir Nolan', 'people.person.gender', 'Male'],
      ['My Worlds: The Collection', 'music.album.genre', 'Teen pop'],
      ["Justin Bieber's Believe", 'film.film.other_crew', 'm.0101ftt1'],
      ['Hot Wired Radio', 'broadcast.content.artist', 'Shiny Toy Guns'],
      ['Synthpop', 'music.genre.parent_genre', 'K-pop'],
      ['Adam Messinger',
       'music.composer.compositions',
       "Turn to You (Mother's Day Dedication)"],
      ['m.0yrktlv',
       'award.award_honor.award',
       'Teen Choice Award for Choice Male Hottie'],
      ['Kanye West', 'people.person.nationality', 'United States of America'],
      ['Iggy Azalea',
       'freebase.valuenotation.is_reviewed',
       'Country of nationality'],
      ["Justin Bieber's Believe", 'film.film.release_date_s', 'm.0101fv4c'],
      ['Juicy J', 'freebase.valuenotation.has_value', 'Parents'],
      ['JellyRadio.com', 'broadcast.content.artist', 'DMX'],
      ['HitzRadio.com', 'broadcast.content.artist', 'The Black Eyed Peas'],
      ['m.0gxnnzy',
       'celebrities.romantic_relationship.relationship_type',
       'Dated'],
      ['Aaliyah', 'broadcast.artist.content', '1Club.FM: Channel One'],
      ['Elvis Presley', 'freebase.valuenotation.is_reviewed', 'Children'],
      ['radioIO Todays POP', 'common.topic.notable_for', 'g.1255g6pyx'],
      ["Justin Bieber's Believe", 'film.film.release_date_s', 'm.0101fvcp'],
      ['m.0njwb81', 'award.award_honor.award', 'UR Fave: New Artist'],
      ['1Club.FM: Channel One', 'broadcast.content.artist', 'Ashlee Simpson'],
      ['L.A. Reid', 'freebase.valuenotation.is_reviewed', 'Gender'],
      ['Britney Spears',
       'broadcast.artist.content',
       "PartyRadioUSA.net - Nation's #1 Party Authority"],
      ['m.0njhxd_', 'freebase.valuenotation.is_reviewed', 'Award winner'],
      ['Michael Jackson', 'broadcast.artist.content', 'radioIO RNB Mix'],
      ['Frank Ocean', 'music.artist.genre', 'Rhythm and blues'],
      ['Ludacris', 'music.artist.contribution', 'm.0vp800w'],
      ['Singer', 'common.topic.subject_of', 'Justin Bieber'],
      ['Fergie', 'music.artist.genre', 'Rock music'],
      ['Gas Pedal', 'common.topic.notable_types', 'Musical Recording'],
      ['Toby Gad', 'people.person.profession', 'Record producer'],
      ['All Around The World', 'music.composition.composer', 'Justin Bieber'],
      ['Mistletoe', 'music.album.release_type', 'Single'],
      ['Kid Cudi', 'people.person.profession', 'Film Producer'],
      ['Hot Wired Radio', 'broadcast.content.artist', 'Ashley Tisdale'],
      ['Somebody to Love (remix)', 'music.album.contributor', 'm.0vp7cl4'],
      ['Live My Life (Party Rock remix)',
       'music.recording.tracks',
       'Live My Life (Party Rock remix)'],
      ['Beauty and a Beat (Bisbetic Instrumental)',
       'music.recording.artist',
       'Justin Bieber'],
      ['m.0njw4z2',
       'award.award_honor.award',
       'MTV Europe Music Award for Best Male'],
      ["Destiny's Child", 'music.artist.genre', 'Contemporary R&B'],
      ['Snoop Dogg', 'people.person.profession', 'Record producer'],
      ['Savan Kotecha', 'music.artist.genre', 'Dance-pop'],
      ['m.0gbm3c3',
       'film.personal_film_appearance.type_of_appearance',
       'Him/Herself'],
      ['Rodney Jerkins', 'people.person.nationality', 'United States of America'],
      ['Justin Bieber', 'broadcast.artist.content', 'Hot Wired Radio'],
      ["PartyRadioUSA.net - Nation's #1 Party Authority",
       'broadcast.content.artist',
       'Miley Cyrus'],
      ['Pop music', 'base.schemastaging.music_genre_concept.artists', 'Yves Bole'],
      ["Destiny's Child", 'music.artist.genre', 'Pop music'],
      ['United States of America',
       'base.biblioness.bibs_topic.is_really',
       'United States of America'],
      ['Christina Aguilera', 'freebase.valuenotation.is_reviewed', 'Parents'],
      ['m.09xx941', 'common.webpage.topic', 'Teen idol'],
      ['Christina Milian', 'people.person.profession', 'Record producer'],
      ['JoJo', 'people.person.nationality', 'United States of America'],
      ['Kylie Minogue', 'music.artist.genre', 'Electronic dance music'],
      ['Next to You', 'music.album.release_type', 'Single'],
      ['#thatPower', 'music.composition.recordings', '#thatPOWER'],
      ['Willa Ford', 'people.person.languages', 'English Language'],
      ['Frank Sinatra', 'freebase.valuenotation.is_reviewed', 'Place of birth'],
      ['All That Matters', 'music.composition.composer', 'Andre Harris'],
      ['Contemporary R&B', 'broadcast.genre.content', 'Smoothbeats'],
      ['Paul Anka', 'music.artist.genre', 'Pop music'],
      ['Geri Halliwell', 'freebase.valuenotation.is_reviewed', 'Date of birth'],
      ['Shaffer Smith', 'broadcast.artist.content', 'Big R Radio - Top 40 Hits'],
      ['Lady Gaga', 'freebase.valuenotation.is_reviewed', 'Gender'],
      ['Jeremy Bieber', 'freebase.valuenotation.has_value', 'Height'],
      ['Caitlin Beadles', 'people.person.nationality', 'Canada'],
      ['m.0z8s_wn', 'award.award_honor.honored_for', 'My World'],
      ['Favorite Girl', 'common.topic.notable_types', 'Musical Album'],
      ['Hot Wired Radio',
       'broadcast.content.broadcast',
       'Hot Wired Radio - 128kbps Stream'],
      ['.977 The Hits Channel', 'broadcast.content.artist', 'R. Kelly'],
      ['Avery', 'common.topic.notable_types', 'Musical Artist'],
      ['m.0gbm3d9',
       'film.personal_film_appearance.film',
       'Justin Bieber: Never Say Never'],
      ['Ernie Isley', 'freebase.valuenotation.is_reviewed', 'Date of birth'],
      ['Beyoncé Knowles', 'people.person.profession', 'Actor'],
      ['m.0yrk18w', 'freebase.valuenotation.has_no_value', 'Winning work'],
      ['Ja Rule', 'freebase.valuenotation.is_reviewed', 'Place of birth'],
      ['Tupac Shakur', 'people.person.profession', 'Actor'],
      ['Stephen Melton', 'common.topic.subjects', 'Singer-songwriter'],
      ['Chris Brown', 'freebase.valuenotation.has_no_value', 'Children'],
      ['Trick Daddy', 'freebase.valuenotation.has_value', 'Parents'],
      ['Diplo', 'freebase.valuenotation.is_reviewed', 'Gender'],
      ['Frank Ocean', 'people.person.nationality', 'United States of America'],
      ['Christina Milian', 'music.composer.compositions', 'Baby'],
      ['Chance the Rapper', 'music.artist.genre', 'Hip hop music'],
      ['Justin Timberlake',
       'freebase.valuenotation.is_reviewed',
       'Country of nationality'],
      ['Khalil', 'people.person.gender', 'Male'],
      ['#thatPOWER', 'music.recording.tracks', '#thatPower (remix)'],
      ['Recovery', 'freebase.valuenotation.is_reviewed', 'Initial release date'],
      ['Selena Gomez',
       'freebase.valuenotation.has_no_value',
       'Spouse (or domestic partner)'],
      ['Juelz Santana', 'broadcast.artist.content', '.977 The Hits Channel'],
      ['Fabolous', 'broadcast.artist.content', 'SoulfulHipHop.com Radio'],
      ['Roller Coaster', 'common.topic.notable_for', 'g.1yp3bnqz7'],
      ['m.0yrk4gn', 'award.award_honor.award_winner', 'Justin Bieber'],
      ["Justin Bieber's Believe", 'film.film.release_date_s', 'm.0101fv7x'],
      ['Jay Cassidy', 'freebase.valuenotation.has_value', 'Parents'],
      ['Anastacia', 'music.artist.genre', 'Contemporary R&B'],
      ['C1', 'music.artist.genre', 'Hip hop music'],
      ['My Worlds Acoustic',
       'freebase.valuenotation.is_reviewed',
       'Album content type'],
      ['m.0bvmhvb', 'common.webpage.resource', 'Justin Bieber Pictures'],
      ['Live My Life', 'music.composition.language', 'English Language'],
      ['Vocals', 'music.instrument.instrumentalists', 'Aaliyah'],
      ['#thatPOWER', 'music.recording.featured_artists', 'Justin Bieber'],
      ['Shorty Award for Celebrity', 'award.award_category.nominees', 'm.0y_g42w'],
      ['Baby', 'music.album.releases', 'Baby'],
      ['A Michael Bublé Christmas', 'common.topic.notable_types', 'Film'],
      ['Right Here', 'music.recording.canonical_version', 'Right Here'],
      ['Justin Bieber', 'people.person.profession', 'Musician'],
      ['50 Cent', 'freebase.valuenotation.is_reviewed', 'Place of birth'],
      ['Bigger', 'music.composition.composer', 'Waynne Nugent'],
      ['Home to Mama', 'music.composition.composer', 'Cody Simpson'],
      ['Big R Radio - The Hawk',
       'broadcast.content.artist',
       'The Black Eyed Peas'],
      ['Thought Of You', 'music.composition.composer', 'Justin Bieber'],
      ['The Black Eyed Peas', 'music.artist.genre', 'Electronic dance music'],
      ['Singer', 'people.profession.specializations', 'Prima donna'],
      ['Alanis Morissette', 'people.person.profession', 'Record producer'],
      ['My World', 'award.award_nominated_work.award_nominations', 'm.0tkc3tj'],
      ['Record producer', 'common.topic.notable_for', 'g.1258k9617'],
      ['Singer', 'base.descriptive_names.names.descriptive_name', 'm.0106bj25'],
      ['Christina Aguilera', 'music.artist.genre', 'Rhythm and blues'],
      ['Mariah Carey', 'broadcast.artist.content', 'SoulfulSmoothJazz.com'],
      ['Justin Bieber: Never Say Never',
       'film.film.production_companies',
       'AEG Live'],
      ['Redfoo', 'people.person.gender', 'Male'],
      ['Chris Brown', 'broadcast.artist.content', '1Club.FM: V101'],
      ['WildFMRadio.com', 'broadcast.content.artist', '50 Cent'],
      ['Ronald Isley', 'music.artist.genre', 'Quiet Storm'],
      ['Nathan Lanier', 'freebase.valuenotation.has_value', 'Parents'],
      ['P!nk', 'freebase.valuenotation.is_reviewed', 'Official website'],
      ['Athan Grace', 'celebrities.celebrity.celebrity_friends', 'm.012r2w0k'],
      ['Miley Cyrus', 'freebase.valuenotation.is_reviewed', 'Profession'],
      ['Right Here', 'music.album.featured_artists', 'Drake'],
      ['m.01053qzf',
       'film.personal_film_appearance.film',
       'Justin Bieber: Never Say Never'],
      ['Guglielmo Scilla', 'common.topic.notable_types', 'Person'],
      ['Justin Bieber', 'award.award_winner.awards_won', 'm.0v90skf'],
      ['Jordan Pruitt', 'music.artist.genre', 'Pop music'],
      ['Mason Levy', 'music.artist.genre', 'Rhythm and blues'],
      ['Thought of You', 'common.topic.notable_types', 'Canonical Version'],
      ['Whitney Houston', 'people.person.profession', 'Record producer'],
      ['m.07lkzw7', 'common.webpage.category', 'Official Website'],
      ['Ray J', 'people.person.profession', 'Musician'],
      ['m.0gbmnvf', 'film.film_crew_gig.film', 'Justin Bieber: Never Say Never'],
      ['Enrique Iglesias', 'people.person.gender', 'Male'],
      ['m.0101fv5f',
       'film.film_regional_release_date.film',
       "Justin Bieber's Believe"],
      ['Somebody to Love', 'music.composition.recordings', 'Somebody to Love'],
      ['HitzRadio.com', 'broadcast.content.artist', 'Nelly'],
      ['Eenie Meenie', 'music.single.versions', 'Eenie Meenie'],
      ['Selena Gomez', 'music.artist.genre', 'Teen pop'],
      ["Justin Bieber's Believe", 'film.film.produced_by', 'Scooter Braun'],
      ['Love Never Felt So Good', 'music.album.genre', 'Disco'],
      ['Tupac Shakur', 'freebase.valuenotation.is_reviewed', 'Parents'],
      ['Justin Bieber: Never Say Never', 'film.film.other_crew', 'm.0gbmntp'],
      ['m.0p85jpp', 'film.personal_film_appearance.person', 'Justin Bieber'],
      ['RedOne', 'freebase.valuenotation.is_reviewed', 'Gender'],
      ['m.0v_729v',
       'tv.tv_guest_personal_appearance.appearance_type',
       'Guest host'],
      ['Janet Jackson', 'freebase.valuenotation.is_reviewed', 'Gender'],
      ['1.FM Top 40', 'broadcast.content.artist', 'Christina Milian'],
      ['Ja Rule', 'music.artist.genre', 'Rhythm and blues'],
      ['Justin Bieber', 'music.featured_artist.albums', 'Runaway Love (remix)'],
      ['RedOne', 'freebase.valuenotation.is_reviewed', 'Official website'],
      ['All Around the World', 'music.recording.featured_artists', 'Ludacris'],
      ['Christina Milian', 'people.person.profession', 'Actor'],
      ['Emphatic Radio.com!', 'broadcast.content.artist', 'The Pussycat Dolls'],
      ['Dance music', 'broadcast.genre.content', '181-party'],
      ['Queen Elizabeth II Diamond Jubilee Medal',
       'award.award_category.winners',
       'm.0njwqrb'],
      ['Sean Kingston', 'people.person.profession', 'Singer'],
      ['DMX', 'broadcast.artist.content', 'Hot 108 Jamz'],
      ['Runaway Love (remix)', 'common.topic.notable_types', 'Musical Recording'],
      ['CMT Music Award: Collaborative Video of the Year',
       'award.award_category.winners',
       'm.0njvs9s'],
      ['m.0yrkr34', 'award.award_honor.award_winner', 'Justin Bieber'],
      ['One Time', 'common.topic.notable_types', 'Musical Album'],
      ['Emphatic Radio.com!', 'broadcast.content.artist', 'Soulja Boy'],
      ['Hot Wired Radio', 'broadcast.content.artist', 'Jupiter Rising'],
      ['Katy Perry', 'music.artist.genre', 'Disco'],
      ['Chingy', 'people.person.profession', 'Actor'],
      ['Eminem', 'freebase.valuenotation.is_reviewed', 'Place of birth'],
      ['The Notorious B.I.G.', 'music.artist.genre', 'Hip hop music'],
      ['Dance music', 'broadcast.genre.content', 'Emphatic Radio.com!'],
      ['Rihanna', 'music.artist.genre', 'Dance-pop'],
      ['Justin Bieber',
       'freebase.valuenotation.is_reviewed',
       'Country of nationality'],
      ['Contemporary R&B', 'common.topic.notable_types', 'Musical genre'],
      ['1Club.FM: Channel One', 'broadcast.content.artist', 'City High'],
      ['Singer', 'base.descriptive_names.names.descriptive_name', 'm.0116rg0f'],
      ['Chingy', 'people.person.gender', 'Male'],
      ['Reed Smoot', 'people.person.gender', 'Male'],
      ["Justin Bieber's Believe", 'film.film.edited_by', 'Jillian Twigger Moul'],
      ['Teyana', 'freebase.valuenotation.has_value', 'Parents'],
      ['Next to You', 'music.recording.song', 'Next to You'],
      ['All Bad', 'music.composition.composer', 'Jason \\"Poo Bear\\" Boyd'],
      ['As Long as You Love Me',
       'music.album.releases',
       'As Long As You Love Me (remixes)'],
      ['Teen Choice Award for Choice Music: Breakout Artist - Male',
       'award.award_category.winners',
       'm.0yrjvlh'],
      ['Justin Bieber', 'award.award_winner.awards_won', 'm.010lkp2z'],
      ['Singer', 'common.topic.article', 'm.09l6h'],
      ['m.012r2w0k', 'celebrities.friendship.friend', 'Justin Bieber'],
      ['Scooter Braun', 'film.producer.film', "Justin Bieber's Believe"],
      ['Justin Bieber: Never Say Never',
       'award.award_winning_work.awards_won',
       'm.0pc670l'],
      ['1Club.FM: Power', 'broadcast.content.artist', 'Jay-Z'],
      ['Beauty And A Beat', 'music.composition.form', 'Song'],
      ['Britney Spears', 'music.artist.genre', 'Electronic dance music'],
      ['HitzRadio.com', 'broadcast.content.artist', "Destiny's Child"],
      ['Beyoncé Knowles',
       'freebase.valuenotation.is_reviewed',
       'Country of nationality'],
      ['Live My Life', 'music.recording.tracks', 'Live My Life'],
      ['m.0njhyh_',
       'award.award_honor.award',
       'Billboard Music Award for Top Streaming Song (Video)'],
      ['Lil Jon', 'freebase.valuenotation.is_reviewed', 'Profession'],
      ['Jeremy Bieber', 'people.person.children', 'Jazmyn Bieber'],
      ['Ludacris', 'people.person.nationality', 'United States of America'],
      ['Justin Bieber: Never Say Never',
       'film.film.film_production_design_by',
       'Devorah Herbert'],
      ['Bryan Adams', 'broadcast.artist.content', '1Club.FM: 80s (Pop)'],
      ['m.0gbmntp', 'film.film_crew_gig.film', 'Justin Bieber: Never Say Never'],
      ['Drake', 'music.artist.genre', 'Rhythm and blues'],
      ['Pattie Mallette', 'base.popstra.organization.supporter', 'm.0gxnp72'],
      ['Nick Jonas',
       'freebase.valuenotation.is_reviewed',
       'Country of nationality'],
      ['justinbieber', 'award.award_winning_work.awards_won', 'm.0z0tmyv'],
      ['Lupe Fiasco',
       'broadcast.artist.content',
       "PartyRadioUSA.net - Nation's #1 Party Authority"],
      ['Martin Kierszenbaum',
       'people.person.place_of_birth',
       'United States of America'],
      ['As Long as You Love Me',
       'music.composition.recordings',
       'As Long as You Love Me'],
      ['Juno Fan Choice Award', 'award.award_category.winners', 'm.0gwhmhm'],
      ['m.0d_hbgr', 'common.webpage.category', 'Lyrics'],
      ['Big Sean', 'music.artist.label', 'The Island Def Jam Music Group'],
      ['Beautiful', 'music.composition.lyricist', 'Toby Gad'],
      ['Redfoo', 'music.artist.genre', 'Electronic dance music'],
      ['1Club.FM: Power', 'broadcast.content.artist', 'Snoop Dogg'],
      ['K-Ci & JoJo', 'broadcast.artist.content', 'Big R Radio - The Hawk'],
      ['Classic Soul Network', 'broadcast.content.genre', 'Contemporary R&B'],
      ['K-Ci & JoJo', 'common.topic.notable_types', 'Musical Artist'],
      ['Stephen Melton', 'music.group_member.instruments_played', 'Vocals'],
      ['SoulfulHipHop.com Radio', 'broadcast.content.genre', 'Rock music'],
      ['Twista', 'freebase.valuenotation.is_reviewed', 'Date of birth'],
      ['Contemporary R&B', 'broadcast.genre.content', '181-thebox'],
      ['Jason Mraz', 'broadcast.artist.content', 'Big R Radio - Top 40 Hits'],
      ['Johntá Austin', 'freebase.valuenotation.has_value', 'Parents'],
      ['m.0y5tl39',
       'film.personal_film_appearance.film',
       'Les Coulisses des Golden Globes'],
      ['Teen idol', 'common.topic.webpage', 'm.09y89l2'],
      ['m.0sgkyfg', 'freebase.valuenotation.has_no_value', 'Winning work'],
      ['Kevin Risto', 'people.person.profession', 'Musician'],
      ['Hot Wired Radio', 'broadcast.content.artist', 'Kings of Leon'],
      ['justinbieber',
       'award.award_nominated_work.award_nominations',
       'm.0z0tgz6'],
      ['Justin Bieber', 'music.artist.label', 'Island Records'],
      ['Ernie Isley', 'people.person.nationality', 'United States of America'],
      ['Kylie Minogue', 'people.person.profession', 'Film Producer'],
      ['Yves Bole', 'tv.tv_actor.starring_roles', 'm.012bm2cn'],
      ['Everlast', 'music.artist.label', 'Island Records'],
      ['5th Annual Shorty Awards',
       'award.award_ceremony.awards_presented',
       'm.0ywvh8k'],
      ['Chance the Rapper', 'music.featured_artist.albums', 'Confident'],
      ['Ludacris', 'freebase.valuenotation.is_reviewed', 'Children'],
      ['Baby', 'common.topic.notable_types', 'Composition'],
      ['Fabian', 'base.icons.icon.icon_genre', 'Teen idol'],
      ['Snoop Dogg', 'broadcast.artist.content', '.977 The Hits Channel'],
      ['m.0tkqqgg',
       'award.award_nomination.award',
       'Juno Award for Pop Album of the Year'],
      ['Ashlee Simpson', 'broadcast.artist.content', '1Club.FM: Channel One'],
      ['Eenie Meenie', 'music.recording.canonical_version', 'Eenie Meenie'],
      ['Person', 'type.type.properties', 'Parents'],
      ['Bryan Adams', 'freebase.valuenotation.is_reviewed', 'Gender'],
      ['Nasri', 'people.person.profession', 'Singer'],
      ['Lady Gaga', 'music.artist.genre', 'Contemporary R&B'],
      ['Vanessa Hudgens', 'broadcast.artist.content', 'Emphatic Radio.com!'],
      ['m.0njhx1b', 'award.award_honor.ceremony', '2011 Billboard Music Awards'],
      ['As Long as You Love Me',
       'music.album.compositions',
       'As Long as You Love Me'],
      ['Madonna', 'broadcast.artist.content', 'Emphatic Radio.com!'],
      ['Demi Lovato', 'freebase.valuenotation.is_reviewed', 'Official website'],
      ['The Black Eyed Peas', 'music.artist.genre', 'Hip hop music'],
      ['Bigger', 'music.composition.composer', 'Frank Ocean'],
      ['Bigger', 'music.composition.recordings', 'Bigger'],
      ['Canadian', 'common.topic.notable_types', 'Ethnicity'],
      ['As Long as You Love Me', 'common.topic.article', 'm.0k0l2vk'],
      ['Musician', 'freebase.equivalent_topic.equivalent_type', 'Musical Artist'],
      ['Jennifer Lopez', 'freebase.valuenotation.is_reviewed', 'Gender'],
      ['Chef Tone', 'people.person.nationality', 'United States of America'],
      ['Whitney Houston', 'music.artist.genre', 'Dance music'],
      ['My Worlds Acoustic', 'music.album.album_content_type', 'Remix album'],
      ['Avery', 'music.artist.label', 'The Island Def Jam Music Group'],
      ['Change Me', 'music.album.primary_release', 'Change Me'],
      ['Nick Jonas', 'base.popstra.celebrity.friendship', 'm.0cq9hwb'],
      ['m.0w3gbtv',
       'film.personal_film_appearance.film',
       'Zendaya: Behind the Scenes'],
      ['Singer', 'base.descriptive_names.names.descriptive_name', 'm.0105_4hw'],
      ['That Should Be Me', 'music.composition.form', 'Song'],
      ['Never Say Never', 'music.album.compositions', 'Never Say Never'],
      ['m.09wsj7g', 'common.webpage.topic', 'Teen idol'],
      ['The Island Def Jam Music Group',
       'music.record_label.artist',
       'Justin Bieber'],
      ['#thatPOWER', 'music.album.releases', '#thatPOWER'],
      ['Ashley Tisdale', 'people.person.profession', 'Actor'],
      ['Sir Nolan', 'music.artist.genre', 'Rock music'],
      ['Beauty and a Beat (acoustic version)',
       'music.recording.song',
       'Beauty And A Beat'],
      ['Ellen DeGeneres', 'people.person.nationality', 'United States of America'],
      ['Sia Furler', 'people.person.profession', 'Singer-songwriter'],
      ['Usher', 'music.composer.compositions', 'First Dance'],
      ['m.0n1ykxp',
       'award.award_honor.award',
       'MTV Video Music Award for Artist to Watch'],
      ['Justin Bieber: Never Say Never',
       'media_common.netflix_title.netflix_genres',
       'Rockumentary'],
      ['Amerie', 'people.person.gender', 'Female'],
      ['Real Change: Artists for Education',
       'film.film.personal_appearances',
       'm.0y5th3r'],
      ['Mistletoe', 'music.album.primary_release', 'Mistletoe'],
      ['Beautiful and the Beat',
       'music.recording.canonical_version',
       'Beauty and a Beat'],
      ['#Thatpower', 'music.recording.tracks', '#thatPOWER'],
      ['Baby', 'common.topic.notable_types', 'Musical Album'],
      ['Big R Radio - The Hawk', 'broadcast.content.artist', 'Flyleaf'],
      ['PYD', 'common.topic.notable_types', 'Composition'],
      ['Ashlee Simpson', 'people.person.profession', 'Singer'],
      ['Pray', 'music.album.artist', 'Justin Bieber'],
      ['Justin Bieber', 'award.award_winner.awards_won', 'm.0z8s562'],
      ['Trey Songz', 'music.artist.genre', 'Contemporary R&B'],
      ['Pras', 'freebase.valuenotation.is_reviewed', 'Date of birth'],
      ['.977 The Hits Channel', 'broadcast.content.artist', 'Coldplay'],
      ['Nicki Minaj', 'freebase.valuenotation.is_reviewed', 'Official website'],
      ['Geri Halliwell', 'people.person.profession', 'Model'],
      ['iJustine', 'people.person.gender', 'Female'],
      ['Nelly Furtado', 'people.person.gender', 'Female'],
      ['Trey Songz', 'people.person.nationality', 'United States of America'],
      ['m.0ng_vkd',
       'film.personal_film_appearance.film',
       'A Michael Bublé Christmas'],
      ["Justin Bieber's Believe", 'film.film.produced_by', "Bill O'Dowd"],
      ['m.0njhtjj', 'freebase.valuenotation.is_reviewed', 'Award winner'],
      ['Ludacris', 'music.composer.compositions', 'Baby'],
      ['Terius Nash', 'music.featured_artist.recordings', 'Baby'],
      ['Ginuwine', 'freebase.valuenotation.is_reviewed', 'Date of birth'],
      ['Somebody to Love', 'common.topic.notable_types', 'Musical Recording'],
      ['Vanessa Hudgens',
       'freebase.valuenotation.is_reviewed',
       'Country of nationality'],
      ['Mary J. Blige', 'music.artist.genre', 'Contemporary R&B'],
      ['Beyoncé Knowles', 'people.person.profession', 'Record producer'],
      ['#thatPOWER', 'music.recording.tracks', '#thatPower'],
      ['m.0z8755b', 'award.award_honor.award_winner', 'Justin Bieber'],
      ['Live My Life', 'common.topic.notable_for', 'g.1yl5pb70b'],
      ['Contemporary R&B', 'broadcast.genre.content', '1Club.FM: V101'],
      ['Big R Radio - Top 40 Hits', 'broadcast.content.artist', 'K-Ci & JoJo'],
      ['CL', 'freebase.valuenotation.is_reviewed', 'Gender'],
      ['Shaggy', 'freebase.valuenotation.is_reviewed', 'Place of birth'],
      ['Gas Pedal', 'music.recording.tracks', 'Gas Pedal'],
      ['Jason Mraz', 'freebase.valuenotation.is_reviewed', 'Profession'],
      ['Beyoncé Knowles', 'broadcast.artist.content', 'Big R Radio - The Hawk'],
      ['Big R Radio - Top 40 Hits', 'broadcast.content.artist', 'Lady Antebellum'],
      ['Ludacris', 'freebase.valuenotation.is_reviewed', 'Gender'],
      ['Madonna', 'people.person.profession', 'Record producer'],
      ['m.0yqfny6', 'freebase.valuenotation.has_no_value', 'Winning work'],
      ['Emphatic Radio.com!', 'broadcast.content.artist', 'Keyshia Cole'],
      ['1Club.FM: Power', 'broadcast.content.genre', 'Hip hop music'],
      ['PowerHitz', 'broadcast.content.artist', 'M.I.A.'],
      ['As Long as You Love Me (acoustic version)',
       'music.recording.song',
       'As Long as You Love Me'],
      ['Shaffer Smith', 'broadcast.artist.content', 'Hot Wired Radio'],
      ['Blu Cantrell', 'people.person.gender', 'Female'],
      ['Contemporary R&B', 'common.topic.notable_for', 'g.125brs11z'],
      ['Rob Thomas', 'people.person.gender', 'Male'],
      ['Singer', 'people.profession.specializations', 'Piano Singer'],
      ['Singer', 'base.descriptive_names.names.descriptive_name', 'm.010b9gzv'],
      ['NME Awards 2011', 'award.award_ceremony.awards_presented', 'm.0z8s_wn'],
      ['m.0hvlt03',
       'film.film_film_distributor_relationship.film',
       'Justin Bieber: Never Say Never'],
      ["Justin Bieber's Believe", 'film.film.release_date_s', 'm.0101fvq6'],
      ['Victoria Justice', 'base.popstra.celebrity.friendship', 'm.0cq9hwb'],
      ['justinbieber',
       'award.award_nominated_work.award_nominations',
       'm.0_srv2b'],
      ['Terence Dudley', 'people.person.profession', 'Musician'],
      ['Donna Summer',
       'freebase.valuenotation.is_reviewed',
       'Country of nationality'],
      ['m.0101fszs',
       'film.personal_film_appearance.film',
       "Justin Bieber's Believe"],
      ['Alanis Morissette',
       'freebase.valuenotation.is_reviewed',
       'Official website'],
      ['1Club.FM: Channel One', 'broadcast.content.artist', 'Lifehouse'],
      ['The Island Def Jam Music Group',
       'music.record_label.artist',
       'Jenna Andrews'],
      ['FLOW 103', 'broadcast.content.artist', 'Cherish'],
      ['Justin Timberlake', 'broadcast.artist.content', '.977 The Hits Channel'],
      ['Next to You', 'music.recording.song', 'Next to You'],
      ['Victoria Justice', 'freebase.valuenotation.is_reviewed', 'Gender'],
      ['Johnny Crawford', 'base.icons.icon.icon_genre', 'Teen idol'],
      ['Ray J', 'people.person.nationality', 'United States of America'],
      ['Usher', 'broadcast.artist.content', 'radioIO RNB Mix'],
      ['Madonna', 'influence.influence_node.influenced', 'Whitney Houston'],
      ['m.0w3gbtv',
       'film.personal_film_appearance.type_of_appearance',
       'Him/Herself'],
      ['Montell Jordan', 'music.artist.genre', 'Hip hop music'],
      ['Nicki Minaj', 'freebase.valuenotation.is_reviewed', 'Parents'],
      ['Fabolous', 'broadcast.artist.content', 'PowerHitz'],
      ['Jessie J', 'freebase.valuenotation.is_reviewed', 'Date of birth'],
      ['Jay-Z', 'common.topic.notable_types', 'Musical Artist'],
      ['Nelly Furtado',
       'freebase.valuenotation.is_reviewed',
       'Country of nationality'],
      ['Max Martin', 'freebase.valuenotation.has_value', 'Parents'],
      ['Record producer', 'common.topic.webpage', 'm.09ygb05'],
      ['As Long As You Love Me (Ferry Corsten remix)',
       'music.recording.canonical_version',
       'As Long As You Love Me'],
      ['Hot Wired Radio', 'broadcast.content.artist', 'Cassie Ventura'],
      ['m.0gbm3fj',
       'film.personal_film_appearance.type_of_appearance',
       'Him/Herself'],
      ['Bryan-Michael Cox',
       'freebase.valuenotation.is_reviewed',
       'Place of birth'],
      ['Big R Radio - Top 40 Hits', 'broadcast.content.artist', 'Juvenile'],
      ['As Long As You Love Me',
       'music.single.versions',
       'As Long As You Love Me (Ferry Corsten club dub)'],
      ['Iggy Azalea', 'music.artist.genre', 'Synthpop'],
      ['Tricky Stewart', 'common.topic.notable_types', 'Record Producer'],
      ['As Long As You Love Me (Ferry Corsten club dub)',
       'common.topic.notable_types',
       'Musical Recording'],
      ['#thatPOWER', 'music.album.album_content_type', 'Studio album'],
      ['50 Cent', 'freebase.valuenotation.is_reviewed', 'Parents'],
      ['Katy Perry', 'music.artist.genre', 'Electronic dance music'],
      ['Kid Cudi', 'people.person.profession', 'Record producer'],
      ['Hot Wired Radio', 'broadcast.content.artist', 'Miley Cyrus'],
      ['m.0wfn4pm', 'people.sibling_relationship.sibling', 'Pattie Mallette'],
      ['Kelly Clarkson', 'freebase.valuenotation.is_reviewed', 'Parents'],
      ['Jaden Smith', 'people.person.profession', 'Dancer'],
      ['m.0z8t2dy', 'award.award_nomination.nominated_for', 'My World'],
      ['Keyshia Cole', 'people.person.profession', 'Record producer'],
      ['Guest host',
       'tv.non_character_role.tv_guest_personal_appearances',
       'm.0v_98y7'],
      ['Person', 'type.type.properties', 'Spouse (or domestic partner)'],
      ['Fall Out Boy', 'music.artist.origin', 'Chicago'],
      ['Jaxon Bieber', 'people.person.sibling_s', 'm.0gxnnwp'],
      ['Mary J. Blige', 'broadcast.artist.content', 'Hot 97.7'],
      ['.977 The Hits Channel', 'broadcast.content.artist', 'Kelly Clarkson'],
      ['FLOW 103', 'broadcast.content.artist', '50 Cent'],
      ['Jordin Sparks', 'music.artist.genre', 'Dance-pop'],
      ['L.A. Reid', 'music.producer.releases_produced', 'My World'],
      ['L.A. Reid', 'people.person.gender', 'Male'],
      ['Jessie J', 'music.artist.genre', 'Hip hop music'],
      ['Big R Radio - Top 40 Hits', 'broadcast.content.artist', 'No Doubt'],
      ['1.FM Top 40', 'broadcast.content.artist', 'Linkin Park'],
      ['Beauty and a Beat (Bisbetic Radio Mix)',
       'music.recording.artist',
       'Justin Bieber'],
      ['London', 'location.location.containedby', 'Ontario'],
      ['Justin Bieber: Never Say Never',
       'film.film.film_set_decoration_by',
       'Lia Roldan'],
      ['Bryan-Michael Cox', 'freebase.valuenotation.is_reviewed', 'Gender'],
      ['Chris Brown', 'music.composer.compositions', 'Next to You'],
      ['Beautiful', 'music.recording.tracks', 'Beautiful'],
      ['Justin Bieber', 'tv.tv_actor.guest_roles', 'm.0gctwjk'],
      ['Children', 'type.property.schema', 'Person'],
      ['Change Me', 'music.album.releases', 'Change Me'],
      ['RedOne', 'music.artist.label', 'Island Records'],
      ['School Gyrls', 'film.film.starring', 'm.0jztshx'],
      ['All Around the World',
       'music.recording.canonical_version',
       'All Around the World'],
      ['m.0y5tl39', 'film.personal_film_appearance.person', 'Justin Bieber'],
      ['Teen Choice Award for Choice Twitter Personality',
       'award.award_category.winners',
       'm.0wjhc6c'],
      ['Live My Life', 'music.recording.featured_artists', 'Justin Bieber'],
      ['Live My Life', 'music.recording.featured_artists', 'Justin Bieber'],
      ['CL', 'freebase.valuenotation.is_reviewed', 'Date of birth'],
      ['Chris Brown', 'broadcast.artist.content', '1Club.FM: Channel One'],
      ['Teen idol', 'base.icons.icon_genre.icons', 'Miley Cyrus'],
      ['m.0z8qqh5', 'award.award_nomination.award_nominee', 'Justin Bieber'],
      ['1.FM Top 40', 'broadcast.content.artist', 'Will Smith'],
      ['Emphatic Radio.com!', 'broadcast.content.artist', 'Baby Bash'],
      ['Adrienne Bailon', 'freebase.valuenotation.is_reviewed', 'Gender'],
      ["PartyRadioUSA.net - Nation's #1 Party Authority",
       'broadcast.content.artist',
       'Lupe Fiasco'],
      ['Hikaru Utada', 'music.artist.label', 'Island Records'],
      ['Dr. Dre', 'people.person.profession', 'Record producer'],
      ['Yves Bole', 'celebrities.celebrity.celebrity_friends', 'm.012bm4v7'],
      ['Carrie Underwood', 'freebase.valuenotation.is_reviewed', 'Profession'],
      ['Shaffer Smith', 'freebase.valuenotation.is_reviewed', 'Gender'],
      ['Justin Bieber', 'music.composer.compositions', 'Change Me'],
      ['Right Here', 'common.topic.notable_types', 'Composition'],
      ['Change Me', 'music.composition.composer', 'Jason \\"Poo Bear\\" Boyd'],
      ['Beauty and a Beat (Wideboys Radio Mix)',
       'music.recording.canonical_version',
       'Beauty and a Beat'],
      ['Madonna', 'freebase.valuenotation.is_reviewed', 'Height'],
      ['#Thatpower', 'music.recording.artist', 'Will i Am'],
      ['Award-Winning Work', 'freebase.type_hints.included_types', 'Topic'],
      ['m.0dm4cqr', 'celebrities.friendship.friend', 'Miley Cyrus'],
      ['Scooter Braun', 'film.producer.film', 'Justin Bieber: Never Say Never'],
      ['Chris Jasper', 'freebase.valuenotation.is_reviewed', 'Place of birth'],
      ['m.0vp7qr5', 'music.recording_contribution.contributor', 'Jaden Smith'],
      ['Eenie Meenie', 'music.recording.artist', 'Sean Kingston'],
      ['m.0v90skf',
       'award.award_honor.award',
       'Billboard Music Award for Top Male Artist'],
      ['Ludacris', 'people.person.profession', 'Actor'],
      ['Heartbreaker', 'music.album.genre', 'Pop music'],
      ['Cameo appearance',
       'tv.special_tv_performance_type.episode_performances',
       'm.0v1lwt2'],
      ['Chef Tone', 'freebase.valuenotation.is_reviewed', 'Date of birth'],
      ['Teen idol', 'common.topic.webpage', 'm.0b47zvy'],
      ['1.FM Top 40', 'broadcast.content.artist', 'Sia Furler'],
      ['Model', 'base.lightweight.profession.similar_professions', 'Actor'],
      ['.977 The Hits Channel', 'broadcast.content.artist', 'Leona Lewis'],
      ['Johntá Austin', 'music.lyricist.lyrics_written', 'Never Let You Go'],
      ['Christina Aguilera', 'broadcast.artist.content', 'Emphatic Radio.com!'],
      ['m.0v_72tb', 'tv.tv_guest_personal_appearance.episode', 'Brown Family'],
      ['The Island Def Jam Music Group',
       'music.record_label.artist',
       'One Chance'],
      ['Never Let You Go', 'common.topic.notable_types', 'Composition'],
      ['Live My Life', 'common.topic.article', 'm.0j4453y'],
      ['Christina Milian', 'freebase.valuenotation.is_reviewed', 'Gender'],
      ["Justin Bieber's Believe", 'film.film.personal_appearances', 'm.0y5t8gm'],
      ['Roller Coaster',
       'award.award_nominated_work.award_nominations',
       'm.0_x4zg3'],
      ['Chris Brown', 'people.person.nationality', 'United States of America'],
      ['Justin Bieber: Never Say Never', 'film.film.produced_by', 'Jane Lipsitz'],
      ['Lupe Fiasco', 'music.artist.genre', 'Hip hop music'],
      ['Teen pop', 'common.topic.article', 'm.02ny8z'],
      ['PowerHitz', 'broadcast.content.genre', 'Contemporary R&B'],
      ['Iggy Azalea', 'people.person.gender', 'Female'],
      ['Sia Furler', 'freebase.valuenotation.is_reviewed', 'Date of birth'],
      ['Adrienne Bailon', 'people.person.profession', 'Dancer'],
      ['Hip hop music', 'broadcast.genre.content', '181-beat'],
      ['m.0sgk_cw',
       'award.award_honor.award',
       "Kids' Choice Award for Favorite Song"],
      ['Ray J', 'freebase.valuenotation.is_reviewed', 'Country of nationality'],
      ['Beyoncé Knowles', 'broadcast.artist.content', 'Sunshine Radio'],
      ['Iggy Azalea', 'music.artist.genre', 'Electronic dance music'],
      ['MTV Video Music Brazil Award for Best International Artist',
       'award.award_category.winners',
       'm.0yrhhqv'],
      ['Mariah Carey', 'music.artist.label', 'Island Records'],
      ['Music', 'common.topic.subject_of', 'POPPMusic.net'],
      ['Camagüey', 'common.topic.notable_types', 'City/Town/Village'],
      ['Enrique Iglesias', 'freebase.valuenotation.is_reviewed', 'Date of birth'],
      ['Favorite Girl', 'music.album.artist', 'Justin Bieber'],
      ['m.0rqp4h0', 'music.track_contribution.track', 'Somebody to Love'],
      ['Britney Spears', 'people.person.profession', 'Singer'],
      ['Die in Your Arms', 'music.recording.song', 'Die in Your Arms'],
      ['Britney Spears', 'freebase.valuenotation.is_reviewed', 'Children'],
      ['Never Say Never', 'common.topic.notable_for', 'g.125bwly1y'],
      ['Miley Cyrus', 'freebase.valuenotation.is_reviewed', 'Parents'],
      ['Rock music',
       'base.webvideo.internet_video_genre.series',
       'Biscuithands, The Animated Musical'],
      ['Chris Brown', 'freebase.valuenotation.is_reviewed', 'Gender'],
      ['Chris Jasper', 'freebase.valuenotation.is_reviewed', 'Gender'],
      ['Chef Tone', 'music.artist.genre', 'Hip hop music'],
      ['Rudolph Isley', 'people.person.gender', 'Male'],
      ['The Island Def Jam Music Group',
       'music.record_label.artist',
       'Barry Weiss'],
      ['Beauty and a Beat (Bisbetic Instrumental)',
       'common.topic.notable_types',
       'Musical Recording'],
      ['MTV Europe Music Award for Best Male',
       'award.award_category.winners',
       'm.0z1scxk'],
      ['Tricky Stewart', 'music.artist.genre', 'Rhythm and blues'],
      ['1Club.FM: Channel One', 'broadcast.content.artist', 'Gwen Stefani'],
      ['Will Smith', 'people.person.profession', 'Actor'],
      ['Yves Bole', 'influence.influence_node.influenced_by', 'iJustine'],
      ['Will i Am',
       'freebase.valuenotation.is_reviewed',
       'Country of nationality'],
      ['Boyfriend', 'music.composition.recordings', 'Boyfriend'],
      ['Selena Gomez', 'freebase.valuenotation.is_reviewed', 'Gender'],
      ['m.0y803nt', 'freebase.valuenotation.is_reviewed', 'Award winner'],
      ['Fabian', 'people.person.gender', 'Male'],
      ['SoulfulHipHop.com Radio', 'broadcast.content.artist', 'Mary J. Blige'],
      ['Somebody to Love (remix)',
       'music.album.primary_release',
       'Somebody to Love (remix)'],
      ['HitzRadio.com', 'broadcast.content.artist', 'Panic! at the Disco'],
      ['Urban contemporary', 'broadcast.genre.content', 'Hot 108 Jamz'],
      ['Eminem', 'freebase.valuenotation.is_reviewed', 'Height'],
      ['#thatPOWER', 'music.single.versions', '#thatPOWER'],
      ['Justin Bieber', 'award.award_winner.awards_won', 'm.0102z0vx'],
      ['Spouse', 'type.property.expected_type', 'Person'],
      ['m.03zb5cw', 'common.webpage.topic', 'HitzRadio.com'],
      ['Baby', 'music.recording.artist', 'Ludacris'],
      ['Rudolph Valentino',
       'people.person.nationality',
       'United States of America'],
      ['Hit-Boy', 'music.artist.genre', 'Hip hop music'],
      ['Judy Garland',
       'freebase.valuenotation.is_reviewed',
       'Country of nationality'],
      ['Kelly Clarkson',
       'freebase.valuenotation.is_reviewed',
       'Country of nationality'],
      ['#thatPower', 'music.composition.recordings', '#Thatpower'],
      ["Justin Bieber's Believe",
       'base.schemastaging.context_name.pronunciation',
       'm.011h9_22'],
      ['.977 The Hits Channel', 'broadcast.content.artist', 'Nelly'],
      ['Miley Cyrus', 'people.person.profession', 'Musician'],
      ['Justin Timberlake', 'people.person.gender', 'Male'],
      ['#Thatpower', 'music.recording.tracks', '#thatPOWER'],
      ['m.0vp8rhw',
       'music.recording_contribution.album',
       'Beauty and a Beat (Remixes)'],
      ['Believe', 'award.award_nominated_work.award_nominations', 'm.0nhfd4m'],
      ['Katy Perry: Part of Me',
       'common.topic.notable_types',
       'Award-Winning Work'],
      ['m.0jsmvv5',
       'film.film_regional_release_date.film',
       'Justin Bieber: Never Say Never'],
      ["Justin Bieber's Believe", 'common.topic.notable_for', 'g.1yj4hbf4k'],
      ['My Worlds: The Collection', 'music.album.release_type', 'Album'],
      ['All Around The World (featuring Ludacris)',
       'music.recording.artist',
       'Justin Bieber'],
      ['Justin Bieber', 'base.popstra.celebrity.hangout', 'm.0gxnp5x'],
      ['1Club.FM: Channel One', 'broadcast.content.artist', 'Lady Gaga'],
      ['1Club.FM: Mix 106', 'broadcast.content.producer', '1Club.FM'],
      ['1Club.FM: Channel One', 'broadcast.content.artist', 'Duffy'],
      ['Big R Radio - The Hawk', 'broadcast.content.artist', 'Dirty Vegas'],
      ['Whitney Houston', 'broadcast.artist.content', 'SoulfulClassics.com'],
      ['Never Let You Go', 'music.composition.lyricist', 'Johntá Austin'],
      ['m.0_x4zg3', 'award.award_nomination.nominated_for', 'Roller Coaster'],
      ['Lady Antebellum', 'common.topic.notable_types', 'Musical Artist'],
      ['School Boy Records', 'music.record_label.artist', 'Madison Beer'],
      ["Justin Bieber's Believe", 'film.film.other_crew', 'm.0101ftl5'],
      ['Musical Album', 'freebase.type_hints.included_types', 'Topic'],
      ['As Long As You Love Me',
       'music.single.versions',
       'As Long As You Love Me (Audien dubstep mix)'],
      ['radioIO Todays RNB', 'broadcast.content.artist', 'Lil Wayne'],
      ['Mary J. Blige', 'broadcast.artist.content', 'radioIO RNB Mix'],
      ['Fergie', 'people.person.profession', 'Actor'],
      ['Demi Lovato', 'freebase.valuenotation.is_reviewed', 'Place of birth'],
      ['Stuart Ford', 'people.person.profession', 'Film Producer'],
      ['Never Let You Go', 'music.composition.composer', 'Bryan-Michael Cox'],
      ['Zac Efron', 'people.person.gender', 'Male'],
      ['P!nk', 'music.artist.genre', 'Rock music'],
      ['R. Kelly', 'people.person.profession', 'Film Producer'],
      ['Gender', 'type.property.schema', 'Person'],
      ['Adam Messinger', 'music.artist.genre', 'Rhythm and blues'],
      ['Selena Gomez', 'influence.influence_node.influenced_by', 'Britney Spears'],
      ['Right Here', 'common.topic.notable_for', 'g.12h31mb_7'],
      ['JoJo', 'broadcast.artist.content', '1Club.FM: Channel One'],
      ['Jessie J', 'influence.influence_node.influenced', 'Yves Bole'],
      ['Under the Mistletoe',
       'freebase.valuenotation.is_reviewed',
       'Initial release date'],
      ['Live My Life', 'music.recording.tracks', 'Live My Life'],
      ['The Island Def Jam Music Group',
       'music.record_label.artist',
       'Slick Rick'],
      ['Amerie', 'music.artist.genre', 'Rock music'],
      ['Mary J. Blige', 'freebase.valuenotation.is_reviewed', 'Gender'],
      ['m.0pbzq13',
       'film.performance.special_performance_type',
       'Cameo appearance'],
      ['Urban contemporary', 'broadcast.genre.content', 'SoulfulHipHop.com Radio'],
      ['Height', 'type.property.unit', 'Meter'],
      ['Iggy Azalea', 'people.person.profession', 'Model'],
      ['NME Awards 2011', 'award.award_ceremony.awards_presented', 'm.0z8s562'],
      ['Ray J',
       'freebase.valuenotation.has_no_value',
       'Spouse (or domestic partner)'],
      ['Yves Bole', 'base.svocab.music_artist.genre', 'Pop'],
      ['Mannie Fresh', 'freebase.valuenotation.is_reviewed', 'Profession'],
      ['Boyfriend (acoustic version)',
       'music.recording.canonical_version',
       'Boyfriend'],
      ['Big Sean', 'freebase.valuenotation.is_reviewed', 'Profession'],
      ['Believe Tour',
       'music.concert_tour.album_or_release_supporting',
       'Believe'],
      ['m.0yrk4gn', 'freebase.valuenotation.has_no_value', 'Winning work'],
      ['Believe Acoustic', 'music.album.release_type', 'Album'],
      ['Diplo', 'freebase.valuenotation.has_value', 'Height'],
      ['Hikaru Utada', 'music.artist.genre', 'Synthpop'],
      ['Roller Coaster', 'music.composition.composer', 'Julian Swirsky'],
      ['Frank Ocean',
       'freebase.valuenotation.is_reviewed',
       'Country of nationality'],
      ['As Long As You Love Me (Audiobot instrumental)',
       'music.recording.song',
       'As Long as You Love Me'],
      ['Elvis Presley', 'music.artist.genre', 'Pop music'],
      ['Lady Gaga', 'music.artist.genre', 'Pop music'],
      ['FLOW 103', 'broadcast.content.artist', 'Shaffer Smith'],
      ['Annette Funicello', 'base.icons.icon.icon_genre', 'Teen idol'],
      ['Usher', 'people.person.nationality', 'United States of America'],
      ['Live My Life', 'music.composition.recordings', 'Live My Life'],
      ['Kelis', 'music.artist.genre', 'Contemporary R&B'],
      ["Justin Bieber's Believe", 'film.film.release_date_s', 'm.0101fv5f'],
      ['Don Henley', 'freebase.valuenotation.is_reviewed', 'Place of birth'],
      ['Next to You', 'music.recording.tracks', 'Next to You'],
      ['m.0gbm3b7',
       'film.personal_film_appearance.type_of_appearance',
       'Him/Herself'],
      ['Twista', 'broadcast.artist.content', '.977 The Hits Channel'],
      ['Sheryl Crow',
       'freebase.valuenotation.is_reviewed',
       'Country of nationality'],
      ['Gwen Stefani', 'broadcast.artist.content', 'Hot Wired Radio'],
      ['All That Matters',
       'music.composition.composer',
       'Jason \\"Poo Bear\\" Boyd'],
      ['Nasri', 'music.artist.genre', 'Reggae'],
      ['#thatPOWER', 'music.recording.song', '#thatPower'],
      ['Beauty and a Beat', 'common.topic.notable_types', 'Musical Album'],
      ['m.0njdns_', 'award.award_honor.ceremony', 'American Music Awards of 2010'],
      ['Yves Bole', 'base.schemastaging.music_artist_extra.genres', 'Europop'],
      ['Bad 25', 'film.film.genre', 'Documentary film'],
      ['Bigger', 'common.topic.image', '2009 Justin Bieber NYC 2'],
      ['Jay-Z', 'broadcast.artist.content', 'radioIO Todays RNB'],
      ['As Long as You Love Me',
       'music.composition.recordings',
       'As Long As You Love Me'],
      ['Fall Out Boy', 'broadcast.artist.content', 'Big R Radio - The Hawk'],
      ['Geri Halliwell', 'people.person.profession', 'Musician'],
      ['Whitney Houston', 'broadcast.artist.content', 'radioIO RNB Mix'],
      ['Bryan-Michael Cox',
       'freebase.valuenotation.is_reviewed',
       'Country of nationality'],
      ['Whitney Houston', 'broadcast.artist.content', 'SoulfulHipHop.com Radio'],
      ['Justin Bieber Videos', 'common.resource.annotations', 'm.0gc_9w6'],
      ['Justin Bieber', 'tv.tv_actor.guest_roles', 'm.0gbcs1_'],
      ['Chris Brown', 'broadcast.artist.content', 'radioIO Todays RNB'],
      ['Coldplay', 'music.artist.genre', 'Rock music'],
      ['Kevin Risto', 'people.person.profession', 'Record producer'],
      ['Whitney Houston', 'people.person.profession', 'Model'],
      ['Demi Lovato', 'freebase.valuenotation.has_no_value', 'Children'],
      ['Coldplay', 'broadcast.artist.content', 'Big R Radio - Top 40 Hits'],
      ['Anastacia', 'freebase.valuenotation.is_reviewed', 'Gender'],
      ['181-beat', 'broadcast.content.artist', 'Cassie Ventura'],
      ['As Long as You Love Me',
       'music.recording.canonical_version',
       'As Long As You Love Me'],
      ['Kylie Minogue', 'freebase.valuenotation.is_reviewed', 'Parents'],
      ['Disney Parks Christmas Day Parade',
       'common.topic.notable_types',
       'Award-Winning Work'],
      ['Ray J', 'people.person.profession', 'Artist'],
      ['Avril Lavigne', 'people.person.profession', 'Singer-songwriter'],
      ['American Music Award for Favorite Pop/Rock Male Artist',
       'award.award_category.winners',
       'm.0ndc0sf'],
      ['Miley Cyrus', 'freebase.valuenotation.is_reviewed', 'Date of birth'],
      ['Music', 'common.topic.subject_of', 'Brian Keith Kennedy'],
      ['The Notorious B.I.G.',
       'freebase.valuenotation.is_reviewed',
       'Place of birth'],
      ['m.0njw1tn', 'freebase.valuenotation.is_reviewed', 'Year'],
      ['Raekwon', 'freebase.valuenotation.is_reviewed', 'Gender'],
      ['Christina Aguilera', 'music.artist.genre', 'Electronic music'],
      ['PowerHitz', 'broadcast.content.artist', 'Outkast'],
      ['U Smile', 'music.music_video.artist', 'Justin Bieber'],
      ['HitzRadio.com', 'broadcast.content.genre', 'Rock music'],
      ['Sean Kingston', 'music.artist.genre', 'Hip hop music'],
      ['Nelly Furtado', 'freebase.valuenotation.is_reviewed', 'Gender'],
      ['Haley James Scott',
       'fictional_universe.fictional_character.occupation',
       'Record producer'],
      ['Kylie Minogue', 'music.artist.genre', 'Rock music'],
      ['Chris Jasper', 'people.person.nationality', 'United States of America'],
      ['Ice Cube', 'freebase.valuenotation.is_reviewed', 'Parents'],
      ['My Worlds: The Collection',
       'music.album.album_content_type',
       'Compilation album'],
      ['Lolly', 'music.album.releases', 'Lolly'],
      ['Toby Gad', 'common.topic.notable_types', 'Record Producer'],
      ['That Should Be Me', 'music.composition.lyricist', 'Adam Messinger'],
      ['1.FM Top 40', 'broadcast.content.artist', 'Gavin DeGraw'],
      ['Big R Radio - Top 40 Hits', 'broadcast.content.artist', 'Sean Combs'],
      ['m.0jvgmxc', 'freebase.valuenotation.has_no_value', 'Winning work'],
      ['Christina Aguilera', 'broadcast.artist.content', '.977 The Hits Channel'],
      ['1Club.FM: Power', 'broadcast.content.artist', 'DMX'],
      ['Ja Rule', 'people.person.profession', 'Singer'],
      ['Singer', 'base.descriptive_names.names.descriptive_name', 'm.0111fg2d'],
      ['Die in Your Arms',
       'award.award_nominated_work.award_nominations',
       'm.0z85qxq'],
      ['Ashley Tisdale', 'people.person.profession', 'Singer-songwriter'],
      ['m.012nv5gz', 'people.place_lived.location', 'Camagüey'],
      ['Kuk Harrell',
       'film.person_or_entity_appearing_in_film.films',
       'm.0101ft5f'],
      ['Somebody to Love (J Stax remix)',
       'music.recording.artist',
       'Justin Bieber'],
      ["Justin Bieber's Believe",
       'film.film.executive_produced_by',
       'Allison Kaye Scarinzi'],
      ['Adam Messinger', 'people.person.nationality', 'Canada'],
      ['Nasri', 'music.artist.genre', 'Pop music'],
      ['#thatPower', 'music.recording.featured_artists', 'Justin Bieber'],
      ['The Island Def Jam Music Group', 'music.record_label.artist', 'Khalil'],
      ['1Club.FM: Power', 'broadcast.content.artist', 'Kid Cudi'],
      ['C1', 'common.topic.notable_types', 'Musical Artist'],
      ['.977 The Hits Channel', 'broadcast.content.artist', 'JoJo'],
      ['School Boy Records', 'freebase.valuenotation.is_reviewed', 'Artists'],
      ['Country', 'freebase.type_profile.strict_included_types', 'Topic'],
      ['Kid Cudi', 'freebase.valuenotation.is_reviewed', 'Place of birth'],
      ['Big Sean', 'music.featured_artist.recordings', 'As Long As You Love Me'],
      ['Tricky Stewart', 'freebase.valuenotation.is_reviewed', 'Gender'],
      ['m.05sp405',
       'organization.organization_relationship.child',
       'Island Records'],
      ['Savan Kotecha', 'people.person.profession', 'Record producer'],
      ['Teen idol', 'base.icons.icon_genre.icons', 'Judy Garland'],
      ['Big R Radio - Top 40 Hits', 'broadcast.content.artist', 'Mariah Carey'],
      ['m.0b47zvy', 'common.webpage.topic', 'Teen idol'],
      ['John Mamann',
       'freebase.valuenotation.is_reviewed',
       'Country of nationality'],
      ['Teen Choice Award for Choice Summer Music Star: Male',
       'award.award_category.winners',
       'm.0yrjynf'],
      ['Juicy J', 'people.person.profession', 'Actor'],
      ['m.0yqflrk',
       'measurement_unit.dated_money_value.source',
       'celebritynetworth.com'],
      ['Miley Cyrus',
       'freebase.valuenotation.is_reviewed',
       'Country of nationality'],
      ['1Club.FM: Power', 'broadcast.content.artist', 'Eminem'],
      ['#thatPOWER', 'common.topic.notable_types', 'Musical Recording'],
      ['m.04q65lb',
       'organization.organization_relationship.child',
       'The Island Def Jam Music Group'],
      ['Big Sean', 'people.person.nationality', 'United States of America'],
      ['Beyoncé Knowles', 'people.person.profession', 'Film Producer'],
      ['R. Kelly', 'broadcast.artist.content', '1Club.FM: V101'],
      ['1.FM Top 40', 'broadcast.content.artist', '\\"Weird Al\\" Yankovic'],
      ['Geri Halliwell', 'people.person.profession', 'Actor'],
      ['Aaliyah', 'broadcast.artist.content', 'Big R Radio - The Hawk'],
      ['My World', 'music.album.artist', 'Justin Bieber'],
      ['Don Henley', 'people.person.gender', 'Male'],
      ['HitzRadio.com', 'broadcast.content.artist', 'Jay-Z'],
      ['Musician', 'people.profession.specializations', 'Singer'],
      ['Die in Your Arms',
       'music.recording.canonical_version',
       'Die in Your Arms'],
      ['Chris Brown', 'broadcast.artist.content', '1Club.FM: Power'],
      ['m.0njvs9s',
       'award.award_honor.award',
       'CMT Music Award: Collaborative Video of the Year'],
      ['Dr. Dre', 'freebase.valuenotation.is_reviewed', 'Parents'],
      ['Justin Bieber',
       'music.artist.album',
       'Turn to You (Mother’s Day Dedication)'],
      ['Ludacris', 'music.artist.contribution', 'm.0vmyv4w'],
      ['Bryan-Michael Cox', 'music.artist.genre', 'Contemporary R&B'],
      ['City/Town/Village',
       'freebase.type_profile.strict_included_types',
       'Topic'],
      ['Recovery', 'common.topic.notable_types', 'Musical Recording'],
      ['Dancer', 'common.topic.notable_types', 'Profession'],
      ['Live My Life', 'common.topic.notable_types', 'Musical Recording'],
      ['Terence Dudley', 'people.person.gender', 'Male'],
      ['Baby', 'music.composition.recordings', 'Polka Face'],
      ['Lil Jon', 'freebase.valuenotation.is_reviewed', 'Date of birth'],
      ['BeirutNights.com Radio',
       'broadcast.content.artist',
       'Mr. Sosa & The Yayo'],
      ['Whitney Houston', 'influence.influence_node.influenced_by', 'Yves Bole'],
      ['Rihanna', 'music.artist.genre', 'Dance music'],
      ['justinbieber', 'common.topic.notable_for', 'g.1yg57rnx6'],
      ['SoulfulSmoothJazz.com', 'broadcast.content.genre', 'Contemporary R&B'],
      ['Gender', 'type.property.expected_type', 'Gender'],
      ['Geri Halliwell', 'freebase.valuenotation.is_reviewed', 'Place of birth'],
      ['m.0101fvbf',
       'film.film_regional_release_date.film',
       "Justin Bieber's Believe"],
      ['m.0yrhrwc',
       'award.award_honor.ceremony',
       '2011 MTV Video Music Aid Japan'],
      ['MTV Europe Music Award for Best North American Act',
       'award.award_category.winners',
       'm.0yrhmll'],
      ['Iggy Azalea', 'freebase.valuenotation.is_reviewed', 'Height'],
      ['m.0101ft1d',
       'film.personal_film_appearance.type_of_appearance',
       'Him/Herself'],
      ['Big R Radio - Top 40 Hits', 'broadcast.content.artist', 'Fuel'],
      ['Singer', 'base.descriptive_names.names.descriptive_name', 'm.0105_3yz'],
      ['Diplo', 'freebase.valuenotation.is_reviewed', 'Date of birth'],
      ['m.0f0dwc4', 'common.webpage.in_index', 'Blissful Master Index'],
      ['Ciara', 'people.person.gender', 'Female'],
      ['Big R Radio - The Hawk', 'broadcast.content.artist', 'Buckcherry'],
      ['Britney Spears', 'music.artist.genre', 'Synthpop'],
      ['Thought of You', 'music.recording.artist', 'Justin Bieber'],
      ['m.0jzrrqs',
       'location.mailing_address.country',
       'United States of America'],
      ['Justin Bieber', 'internet.blogger.blog', 'justinbieber'],
      ['Live My Life', 'music.composition.recordings', 'Live My Life'],
      ['Toby Gad', 'people.person.nationality', 'United States of America'],
      ['Big R Radio - Top 40 Hits',
       'broadcast.content.artist',
       'Natasha Bedingfield'],
      ['Hot Wired Radio', 'broadcast.content.genre', 'Rock music'],
      ...],
     'choices': []}



Although this dataset can be trained on as-is, a couple problems emerge
from doing so: 1. A retrieval algorithm needs to be implemented and
executed during inference time, that might not appropriately correspond
to the algorithm that was used to generate the dataset subgraphs. 2. The
dataset as is not stored computationally efficiently, as there will
exist many duplicate nodes and edges that are shared between the
questions.

As a result, it makes sense in this scenario to be able to encode all
the entries into a large knowledge graph, so that duplicate nodes and
edges can be avoided, and so that alternative retrieval algorithms can
be tried. We can do this with the LargeGraphIndexer class:

.. code:: ipython3

    from torch_geometric.data import LargeGraphIndexer, Data, get_features_for_triplets_groups
    from torch_geometric.nn.nlp import SentenceTransformer
    import time
    import torch
    import tqdm
    from itertools import chain
    import networkx as nx

.. code:: ipython3

    raw_dataset_graphs = [[tuple(trip) for trip in graph] for graph in ds.raw_dataset['graph']]
    print(raw_dataset_graphs[0][:10])


.. parsed-literal::

    [('P!nk', 'freebase.valuenotation.is_reviewed', 'Gender'), ('1Club.FM: Power', 'broadcast.content.artist', 'P!nk'), ('Somebody to Love', 'music.recording.contributions', 'm.0rqp4h0'), ('Rudolph Valentino', 'freebase.valuenotation.is_reviewed', 'Place of birth'), ('Ice Cube', 'broadcast.artist.content', '.977 The Hits Channel'), ('Colbie Caillat', 'broadcast.artist.content', 'Hot Wired Radio'), ('Stephen Melton', 'people.person.nationality', 'United States of America'), ('Record producer', 'music.performance_role.regular_performances', 'm.012m1vf1'), ('Justin Bieber', 'award.award_winner.awards_won', 'm.0yrkc0l'), ('1.FM Top 40', 'broadcast.content.artist', 'Geri Halliwell')]


To show the benefits of this indexer in action, we will use the
following model to encode this sample of graphs using LargeGraphIndexer,
along with naively.

.. code:: ipython3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(model_name='sentence-transformers/all-roberta-large-v1').to(device)
    print(device)


.. parsed-literal::

    cuda


First, we compare the clock times of encoding using both methods.

.. code:: ipython3

    # Indexing question-by-question
    dataset_graphs_embedded = []
    start = time.time()
    for graph in tqdm.tqdm(raw_dataset_graphs):
        nodes_map = dict()
        edges_map = dict()
        edge_idx_base = []
    
        for src, edge, dst in graph:
            # Collect nodes
            if src not in nodes_map:
                nodes_map[src] = len(nodes_map)
            if dst not in nodes_map:
                nodes_map[dst] = len(nodes_map)
            
            # Collect edge types
            if edge not in edges_map:
                edges_map[edge] = len(edges_map)
    
            # Record edge
            edge_idx_base.append((nodes_map[src], edges_map[edge], nodes_map[dst]))
        
        # Encode nodes and edges
        sorted_nodes = list(sorted(nodes_map.keys(), key=lambda x: nodes_map[x]))
        sorted_edges = list(sorted(edges_map.keys(), key=lambda x: edges_map[x]))
    
        x = model.encode(sorted_nodes, batch_size=256)
        edge_attrs_map = model.encode(sorted_edges, batch_size=256)
        
        edge_attrs = []
        edge_idx = []
        for trip in edge_idx_base:
            edge_attrs.append(edge_attrs_map[trip[1]])
            edge_idx.append([trip[0], trip[2]])
    
        dataset_graphs_embedded.append(Data(x=x, edge_index=torch.tensor(edge_idx).T, edge_attr=torch.stack(edge_attrs, dim=0)))
        
        
    print(time.time()-start)


.. parsed-literal::

    100%|██████████| 100/100 [02:01<00:00,  1.22s/it]

.. parsed-literal::

    121.68579435348511


.. parsed-literal::

    


.. code:: ipython3

    # Using LargeGraphIndexer to make one large knowledge graph
    from torch_geometric.data.large_graph_indexer import EDGE_RELATION
    
    start = time.time()
    all_triplets_together = chain.from_iterable(raw_dataset_graphs)
    # Index as one large graph
    print('Indexing...')
    indexer = LargeGraphIndexer.from_triplets(all_triplets_together)
    
    # first the nodes
    unique_nodes = indexer.get_unique_node_features()
    node_encs = model.encode(unique_nodes, batch_size=256)
    indexer.add_node_feature(new_feature_name='x', new_feature_vals=node_encs)
    
    # then the edges
    unique_edges = indexer.get_unique_edge_features(feature_name=EDGE_RELATION)
    edge_attr = model.encode(unique_edges, batch_size=256)
    indexer.add_edge_feature(new_feature_name="edge_attr", new_feature_vals=edge_attr, map_from_feature=EDGE_RELATION)
    
    ckpt_time = time.time()
    whole_knowledge_graph = indexer.to_data(node_feature_name='x', edge_feature_name='edge_attr') 
    whole_graph_done = time.time()
    print(f"Time to create whole knowledge_graph: {whole_graph_done-start}")
    
    # Compute this to make sure we're comparing like to like on final time printout
    whole_graph_diff = whole_graph_done-ckpt_time
    
    # retrieve subgraphs
    print('Retrieving Subgraphs...')
    dataset_graphs_embedded_largegraphindexer = [graph for graph in tqdm.tqdm(get_features_for_triplets_groups(indexer=indexer, triplet_groups=raw_dataset_graphs), total=num_questions)]
    print(time.time()-start-whole_graph_diff)


.. parsed-literal::

    Indexing...
    Time to create whole knowledge_graph: 114.01080107688904
    Retrieving Subgraphs...


.. parsed-literal::

    100%|██████████| 100/100 [00:00<00:00, 212.87it/s]
    100%|██████████| 100/100 [00:01<00:00, 80.90it/s]

.. parsed-literal::

    114.66037964820862


.. parsed-literal::

    


The large graph indexer allows us to compute the entire knowledge graph
from a series of samples, so that new retrieval methods can also be
tested on the entire graph. We will see this attempted in practice later
on.

It’s worth noting that, although the times are relatively similar right
now, the speedup with largegraphindexer will be much higher as the size
of the knowledge graph grows. This is due to the speedup being a factor
of the number of unique nodes and edges in the graph.

.. code:: ipython3

    dataset_graphs_embedded_largegraphindexer




.. parsed-literal::

    [Data(x=[1723, 1024], edge_index=[2, 9088], edge_attr=[9088, 1024], pid=[100], e_pid=[100], node_idx=[1723], edge_idx=[9088]),
     Data(x=[1253, 1024], edge_index=[2, 4135], edge_attr=[4135, 1024], pid=[100], e_pid=[100], node_idx=[1253], edge_idx=[4135]),
     Data(x=[1286, 1024], edge_index=[2, 2174], edge_attr=[2174, 1024], pid=[100], e_pid=[100], node_idx=[1286], edge_idx=[2174]),
     Data(x=[1988, 1024], edge_index=[2, 5734], edge_attr=[5734, 1024], pid=[100], e_pid=[100], node_idx=[1988], edge_idx=[5734]),
     Data(x=[633, 1024], edge_index=[2, 1490], edge_attr=[1490, 1024], pid=[100], e_pid=[100], node_idx=[633], edge_idx=[1490]),
     Data(x=[1047, 1024], edge_index=[2, 2772], edge_attr=[2772, 1024], pid=[100], e_pid=[100], node_idx=[1047], edge_idx=[2772]),
     Data(x=[1383, 1024], edge_index=[2, 3987], edge_attr=[3987, 1024], pid=[100], e_pid=[100], node_idx=[1383], edge_idx=[3987]),
     Data(x=[1064, 1024], edge_index=[2, 2456], edge_attr=[2456, 1024], pid=[100], e_pid=[100], node_idx=[1064], edge_idx=[2456]),
     Data(x=[1030, 1024], edge_index=[2, 4162], edge_attr=[4162, 1024], pid=[100], e_pid=[100], node_idx=[1030], edge_idx=[4162]),
     Data(x=[1979, 1024], edge_index=[2, 6540], edge_attr=[6540, 1024], pid=[100], e_pid=[100], node_idx=[1979], edge_idx=[6540]),
     Data(x=[1952, 1024], edge_index=[2, 5357], edge_attr=[5357, 1024], pid=[100], e_pid=[100], node_idx=[1952], edge_idx=[5357]),
     Data(x=[1900, 1024], edge_index=[2, 5871], edge_attr=[5871, 1024], pid=[100], e_pid=[100], node_idx=[1900], edge_idx=[5871]),
     Data(x=[1066, 1024], edge_index=[2, 3459], edge_attr=[3459, 1024], pid=[100], e_pid=[100], node_idx=[1066], edge_idx=[3459]),
     Data(x=[1509, 1024], edge_index=[2, 4056], edge_attr=[4056, 1024], pid=[100], e_pid=[100], node_idx=[1509], edge_idx=[4056]),
     Data(x=[2000, 1024], edge_index=[2, 4955], edge_attr=[4955, 1024], pid=[100], e_pid=[100], node_idx=[2000], edge_idx=[4955]),
     Data(x=[1979, 1024], edge_index=[2, 4810], edge_attr=[4810, 1024], pid=[100], e_pid=[100], node_idx=[1979], edge_idx=[4810]),
     Data(x=[1531, 1024], edge_index=[2, 5509], edge_attr=[5509, 1024], pid=[100], e_pid=[100], node_idx=[1531], edge_idx=[5509]),
     Data(x=[1986, 1024], edge_index=[2, 6926], edge_attr=[6926, 1024], pid=[100], e_pid=[100], node_idx=[1986], edge_idx=[6926]),
     Data(x=[574, 1024], edge_index=[2, 1664], edge_attr=[1664, 1024], pid=[100], e_pid=[100], node_idx=[574], edge_idx=[1664]),
     Data(x=[690, 1024], edge_index=[2, 2167], edge_attr=[2167, 1024], pid=[100], e_pid=[100], node_idx=[690], edge_idx=[2167]),
     Data(x=[1425, 1024], edge_index=[2, 3985], edge_attr=[3985, 1024], pid=[100], e_pid=[100], node_idx=[1425], edge_idx=[3985]),
     Data(x=[851, 1024], edge_index=[2, 1934], edge_attr=[1934, 1024], pid=[100], e_pid=[100], node_idx=[851], edge_idx=[1934]),
     Data(x=[1618, 1024], edge_index=[2, 5270], edge_attr=[5270, 1024], pid=[100], e_pid=[100], node_idx=[1618], edge_idx=[5270]),
     Data(x=[1992, 1024], edge_index=[2, 7068], edge_attr=[7068, 1024], pid=[100], e_pid=[100], node_idx=[1992], edge_idx=[7068]),
     Data(x=[1994, 1024], edge_index=[2, 4415], edge_attr=[4415, 1024], pid=[100], e_pid=[100], node_idx=[1994], edge_idx=[4415]),
     Data(x=[1996, 1024], edge_index=[2, 6744], edge_attr=[6744, 1024], pid=[100], e_pid=[100], node_idx=[1996], edge_idx=[6744]),
     Data(x=[656, 1024], edge_index=[2, 1297], edge_attr=[1297, 1024], pid=[100], e_pid=[100], node_idx=[656], edge_idx=[1297]),
     Data(x=[881, 1024], edge_index=[2, 2168], edge_attr=[2168, 1024], pid=[100], e_pid=[100], node_idx=[881], edge_idx=[2168]),
     Data(x=[756, 1024], edge_index=[2, 1539], edge_attr=[1539, 1024], pid=[100], e_pid=[100], node_idx=[756], edge_idx=[1539]),
     Data(x=[1864, 1024], edge_index=[2, 8061], edge_attr=[8061, 1024], pid=[100], e_pid=[100], node_idx=[1864], edge_idx=[8061]),
     Data(x=[1895, 1024], edge_index=[2, 5865], edge_attr=[5865, 1024], pid=[100], e_pid=[100], node_idx=[1895], edge_idx=[5865]),
     Data(x=[873, 1024], edge_index=[2, 3519], edge_attr=[3519, 1024], pid=[100], e_pid=[100], node_idx=[873], edge_idx=[3519]),
     Data(x=[1816, 1024], edge_index=[2, 6375], edge_attr=[6375, 1024], pid=[100], e_pid=[100], node_idx=[1816], edge_idx=[6375]),
     Data(x=[786, 1024], edge_index=[2, 1901], edge_attr=[1901, 1024], pid=[100], e_pid=[100], node_idx=[786], edge_idx=[1901]),
     Data(x=[885, 1024], edge_index=[2, 2366], edge_attr=[2366, 1024], pid=[100], e_pid=[100], node_idx=[885], edge_idx=[2366]),
     Data(x=[1228, 1024], edge_index=[2, 2634], edge_attr=[2634, 1024], pid=[100], e_pid=[100], node_idx=[1228], edge_idx=[2634]),
     Data(x=[1358, 1024], edge_index=[2, 3451], edge_attr=[3451, 1024], pid=[100], e_pid=[100], node_idx=[1358], edge_idx=[3451]),
     Data(x=[1367, 1024], edge_index=[2, 3654], edge_attr=[3654, 1024], pid=[100], e_pid=[100], node_idx=[1367], edge_idx=[3654]),
     Data(x=[977, 1024], edge_index=[2, 2903], edge_attr=[2903, 1024], pid=[100], e_pid=[100], node_idx=[977], edge_idx=[2903]),
     Data(x=[1401, 1024], edge_index=[2, 4570], edge_attr=[4570, 1024], pid=[100], e_pid=[100], node_idx=[1401], edge_idx=[4570]),
     Data(x=[1168, 1024], edge_index=[2, 4004], edge_attr=[4004, 1024], pid=[100], e_pid=[100], node_idx=[1168], edge_idx=[4004]),
     Data(x=[1956, 1024], edge_index=[2, 8173], edge_attr=[8173, 1024], pid=[100], e_pid=[100], node_idx=[1956], edge_idx=[8173]),
     Data(x=[1259, 1024], edge_index=[2, 4246], edge_attr=[4246, 1024], pid=[100], e_pid=[100], node_idx=[1259], edge_idx=[4246]),
     Data(x=[1536, 1024], edge_index=[2, 8149], edge_attr=[8149, 1024], pid=[100], e_pid=[100], node_idx=[1536], edge_idx=[8149]),
     Data(x=[1981, 1024], edge_index=[2, 6006], edge_attr=[6006, 1024], pid=[100], e_pid=[100], node_idx=[1981], edge_idx=[6006]),
     Data(x=[1119, 1024], edge_index=[2, 4501], edge_attr=[4501, 1024], pid=[100], e_pid=[100], node_idx=[1119], edge_idx=[4501]),
     Data(x=[1395, 1024], edge_index=[2, 7217], edge_attr=[7217, 1024], pid=[100], e_pid=[100], node_idx=[1395], edge_idx=[7217]),
     Data(x=[983, 1024], edge_index=[2, 2642], edge_attr=[2642, 1024], pid=[100], e_pid=[100], node_idx=[983], edge_idx=[2642]),
     Data(x=[1634, 1024], edge_index=[2, 3905], edge_attr=[3905, 1024], pid=[100], e_pid=[100], node_idx=[1634], edge_idx=[3905]),
     Data(x=[1182, 1024], edge_index=[2, 3135], edge_attr=[3135, 1024], pid=[100], e_pid=[100], node_idx=[1182], edge_idx=[3135]),
     Data(x=[703, 1024], edge_index=[2, 1575], edge_attr=[1575, 1024], pid=[100], e_pid=[100], node_idx=[703], edge_idx=[1575]),
     Data(x=[194, 1024], edge_index=[2, 428], edge_attr=[428, 1024], pid=[100], e_pid=[100], node_idx=[194], edge_idx=[428]),
     Data(x=[876, 1024], edge_index=[2, 4971], edge_attr=[4971, 1024], pid=[100], e_pid=[100], node_idx=[876], edge_idx=[4971]),
     Data(x=[1964, 1024], edge_index=[2, 7721], edge_attr=[7721, 1024], pid=[100], e_pid=[100], node_idx=[1964], edge_idx=[7721]),
     Data(x=[1956, 1024], edge_index=[2, 5400], edge_attr=[5400, 1024], pid=[100], e_pid=[100], node_idx=[1956], edge_idx=[5400]),
     Data(x=[1918, 1024], edge_index=[2, 6171], edge_attr=[6171, 1024], pid=[100], e_pid=[100], node_idx=[1918], edge_idx=[6171]),
     Data(x=[1351, 1024], edge_index=[2, 3741], edge_attr=[3741, 1024], pid=[100], e_pid=[100], node_idx=[1351], edge_idx=[3741]),
     Data(x=[475, 1024], edge_index=[2, 1488], edge_attr=[1488, 1024], pid=[100], e_pid=[100], node_idx=[475], edge_idx=[1488]),
     Data(x=[1990, 1024], edge_index=[2, 5011], edge_attr=[5011, 1024], pid=[100], e_pid=[100], node_idx=[1990], edge_idx=[5011]),
     Data(x=[509, 1024], edge_index=[2, 986], edge_attr=[986, 1024], pid=[100], e_pid=[100], node_idx=[509], edge_idx=[986]),
     Data(x=[943, 1024], edge_index=[2, 2569], edge_attr=[2569, 1024], pid=[100], e_pid=[100], node_idx=[943], edge_idx=[2569]),
     Data(x=[739, 1024], edge_index=[2, 2404], edge_attr=[2404, 1024], pid=[100], e_pid=[100], node_idx=[739], edge_idx=[2404]),
     Data(x=[1674, 1024], edge_index=[2, 8595], edge_attr=[8595, 1024], pid=[100], e_pid=[100], node_idx=[1674], edge_idx=[8595]),
     Data(x=[1998, 1024], edge_index=[2, 5444], edge_attr=[5444, 1024], pid=[100], e_pid=[100], node_idx=[1998], edge_idx=[5444]),
     Data(x=[1223, 1024], edge_index=[2, 5361], edge_attr=[5361, 1024], pid=[100], e_pid=[100], node_idx=[1223], edge_idx=[5361]),
     Data(x=[428, 1024], edge_index=[2, 1377], edge_attr=[1377, 1024], pid=[100], e_pid=[100], node_idx=[428], edge_idx=[1377]),
     Data(x=[1767, 1024], edge_index=[2, 4428], edge_attr=[4428, 1024], pid=[100], e_pid=[100], node_idx=[1767], edge_idx=[4428]),
     Data(x=[404, 1024], edge_index=[2, 734], edge_attr=[734, 1024], pid=[100], e_pid=[100], node_idx=[404], edge_idx=[734]),
     Data(x=[1416, 1024], edge_index=[2, 4094], edge_attr=[4094, 1024], pid=[100], e_pid=[100], node_idx=[1416], edge_idx=[4094]),
     Data(x=[1658, 1024], edge_index=[2, 6257], edge_attr=[6257, 1024], pid=[100], e_pid=[100], node_idx=[1658], edge_idx=[6257]),
     Data(x=[1907, 1024], edge_index=[2, 7995], edge_attr=[7995, 1024], pid=[100], e_pid=[100], node_idx=[1907], edge_idx=[7995]),
     Data(x=[1992, 1024], edge_index=[2, 4590], edge_attr=[4590, 1024], pid=[100], e_pid=[100], node_idx=[1992], edge_idx=[4590]),
     Data(x=[645, 1024], edge_index=[2, 1666], edge_attr=[1666, 1024], pid=[100], e_pid=[100], node_idx=[645], edge_idx=[1666]),
     Data(x=[1867, 1024], edge_index=[2, 4828], edge_attr=[4828, 1024], pid=[100], e_pid=[100], node_idx=[1867], edge_idx=[4828]),
     Data(x=[1998, 1024], edge_index=[2, 5556], edge_attr=[5556, 1024], pid=[100], e_pid=[100], node_idx=[1998], edge_idx=[5556]),
     Data(x=[1026, 1024], edge_index=[2, 3280], edge_attr=[3280, 1024], pid=[100], e_pid=[100], node_idx=[1026], edge_idx=[3280]),
     Data(x=[1956, 1024], edge_index=[2, 7203], edge_attr=[7203, 1024], pid=[100], e_pid=[100], node_idx=[1956], edge_idx=[7203]),
     Data(x=[1986, 1024], edge_index=[2, 6926], edge_attr=[6926, 1024], pid=[100], e_pid=[100], node_idx=[1986], edge_idx=[6926]),
     Data(x=[836, 1024], edge_index=[2, 1527], edge_attr=[1527, 1024], pid=[100], e_pid=[100], node_idx=[836], edge_idx=[1527]),
     Data(x=[1367, 1024], edge_index=[2, 3654], edge_attr=[3654, 1024], pid=[100], e_pid=[100], node_idx=[1367], edge_idx=[3654]),
     Data(x=[1695, 1024], edge_index=[2, 5494], edge_attr=[5494, 1024], pid=[100], e_pid=[100], node_idx=[1695], edge_idx=[5494]),
     Data(x=[371, 1024], edge_index=[2, 722], edge_attr=[722, 1024], pid=[100], e_pid=[100], node_idx=[371], edge_idx=[722]),
     Data(x=[1986, 1024], edge_index=[2, 6049], edge_attr=[6049, 1024], pid=[100], e_pid=[100], node_idx=[1986], edge_idx=[6049]),
     Data(x=[815, 1024], edge_index=[2, 2322], edge_attr=[2322, 1024], pid=[100], e_pid=[100], node_idx=[815], edge_idx=[2322]),
     Data(x=[1026, 1024], edge_index=[2, 3285], edge_attr=[3285, 1024], pid=[100], e_pid=[100], node_idx=[1026], edge_idx=[3285]),
     Data(x=[1233, 1024], edge_index=[2, 3088], edge_attr=[3088, 1024], pid=[100], e_pid=[100], node_idx=[1233], edge_idx=[3088]),
     Data(x=[290, 1024], edge_index=[2, 577], edge_attr=[577, 1024], pid=[100], e_pid=[100], node_idx=[290], edge_idx=[577]),
     Data(x=[1358, 1024], edge_index=[2, 4891], edge_attr=[4891, 1024], pid=[100], e_pid=[100], node_idx=[1358], edge_idx=[4891]),
     Data(x=[1946, 1024], edge_index=[2, 6642], edge_attr=[6642, 1024], pid=[100], e_pid=[100], node_idx=[1946], edge_idx=[6642]),
     Data(x=[406, 1024], edge_index=[2, 1000], edge_attr=[1000, 1024], pid=[100], e_pid=[100], node_idx=[406], edge_idx=[1000]),
     Data(x=[1973, 1024], edge_index=[2, 5091], edge_attr=[5091, 1024], pid=[100], e_pid=[100], node_idx=[1973], edge_idx=[5091]),
     Data(x=[1124, 1024], edge_index=[2, 4301], edge_attr=[4301, 1024], pid=[100], e_pid=[100], node_idx=[1124], edge_idx=[4301]),
     Data(x=[1530, 1024], edge_index=[2, 4502], edge_attr=[4502, 1024], pid=[100], e_pid=[100], node_idx=[1530], edge_idx=[4502]),
     Data(x=[1020, 1024], edge_index=[2, 2425], edge_attr=[2425, 1024], pid=[100], e_pid=[100], node_idx=[1020], edge_idx=[2425]),
     Data(x=[1410, 1024], edge_index=[2, 8048], edge_attr=[8048, 1024], pid=[100], e_pid=[100], node_idx=[1410], edge_idx=[8048]),
     Data(x=[1998, 1024], edge_index=[2, 5556], edge_attr=[5556, 1024], pid=[100], e_pid=[100], node_idx=[1998], edge_idx=[5556]),
     Data(x=[1996, 1024], edge_index=[2, 4360], edge_attr=[4360, 1024], pid=[100], e_pid=[100], node_idx=[1996], edge_idx=[4360]),
     Data(x=[1867, 1024], edge_index=[2, 4828], edge_attr=[4828, 1024], pid=[100], e_pid=[100], node_idx=[1867], edge_idx=[4828]),
     Data(x=[1866, 1024], edge_index=[2, 5171], edge_attr=[5171, 1024], pid=[100], e_pid=[100], node_idx=[1866], edge_idx=[5171]),
     Data(x=[293, 1024], edge_index=[2, 422], edge_attr=[422, 1024], pid=[100], e_pid=[100], node_idx=[293], edge_idx=[422])]



.. code:: ipython3

    dataset_graphs_embedded




.. parsed-literal::

    [Data(x=[1723, 1024], edge_index=[2, 9088], edge_attr=[9088, 1024]),
     Data(x=[1253, 1024], edge_index=[2, 4135], edge_attr=[4135, 1024]),
     Data(x=[1286, 1024], edge_index=[2, 2174], edge_attr=[2174, 1024]),
     Data(x=[1988, 1024], edge_index=[2, 5734], edge_attr=[5734, 1024]),
     Data(x=[633, 1024], edge_index=[2, 1490], edge_attr=[1490, 1024]),
     Data(x=[1047, 1024], edge_index=[2, 2772], edge_attr=[2772, 1024]),
     Data(x=[1383, 1024], edge_index=[2, 3987], edge_attr=[3987, 1024]),
     Data(x=[1064, 1024], edge_index=[2, 2456], edge_attr=[2456, 1024]),
     Data(x=[1030, 1024], edge_index=[2, 4162], edge_attr=[4162, 1024]),
     Data(x=[1979, 1024], edge_index=[2, 6540], edge_attr=[6540, 1024]),
     Data(x=[1952, 1024], edge_index=[2, 5357], edge_attr=[5357, 1024]),
     Data(x=[1900, 1024], edge_index=[2, 5871], edge_attr=[5871, 1024]),
     Data(x=[1066, 1024], edge_index=[2, 3459], edge_attr=[3459, 1024]),
     Data(x=[1509, 1024], edge_index=[2, 4056], edge_attr=[4056, 1024]),
     Data(x=[2000, 1024], edge_index=[2, 4955], edge_attr=[4955, 1024]),
     Data(x=[1979, 1024], edge_index=[2, 4810], edge_attr=[4810, 1024]),
     Data(x=[1531, 1024], edge_index=[2, 5509], edge_attr=[5509, 1024]),
     Data(x=[1986, 1024], edge_index=[2, 6926], edge_attr=[6926, 1024]),
     Data(x=[574, 1024], edge_index=[2, 1664], edge_attr=[1664, 1024]),
     Data(x=[690, 1024], edge_index=[2, 2167], edge_attr=[2167, 1024]),
     Data(x=[1425, 1024], edge_index=[2, 3985], edge_attr=[3985, 1024]),
     Data(x=[851, 1024], edge_index=[2, 1934], edge_attr=[1934, 1024]),
     Data(x=[1618, 1024], edge_index=[2, 5270], edge_attr=[5270, 1024]),
     Data(x=[1992, 1024], edge_index=[2, 7068], edge_attr=[7068, 1024]),
     Data(x=[1994, 1024], edge_index=[2, 4415], edge_attr=[4415, 1024]),
     Data(x=[1996, 1024], edge_index=[2, 6744], edge_attr=[6744, 1024]),
     Data(x=[656, 1024], edge_index=[2, 1297], edge_attr=[1297, 1024]),
     Data(x=[881, 1024], edge_index=[2, 2168], edge_attr=[2168, 1024]),
     Data(x=[756, 1024], edge_index=[2, 1539], edge_attr=[1539, 1024]),
     Data(x=[1864, 1024], edge_index=[2, 8061], edge_attr=[8061, 1024]),
     Data(x=[1895, 1024], edge_index=[2, 5865], edge_attr=[5865, 1024]),
     Data(x=[873, 1024], edge_index=[2, 3519], edge_attr=[3519, 1024]),
     Data(x=[1816, 1024], edge_index=[2, 6375], edge_attr=[6375, 1024]),
     Data(x=[786, 1024], edge_index=[2, 1901], edge_attr=[1901, 1024]),
     Data(x=[885, 1024], edge_index=[2, 2366], edge_attr=[2366, 1024]),
     Data(x=[1228, 1024], edge_index=[2, 2634], edge_attr=[2634, 1024]),
     Data(x=[1358, 1024], edge_index=[2, 3451], edge_attr=[3451, 1024]),
     Data(x=[1367, 1024], edge_index=[2, 3654], edge_attr=[3654, 1024]),
     Data(x=[977, 1024], edge_index=[2, 2903], edge_attr=[2903, 1024]),
     Data(x=[1401, 1024], edge_index=[2, 4570], edge_attr=[4570, 1024]),
     Data(x=[1168, 1024], edge_index=[2, 4004], edge_attr=[4004, 1024]),
     Data(x=[1956, 1024], edge_index=[2, 8173], edge_attr=[8173, 1024]),
     Data(x=[1259, 1024], edge_index=[2, 4246], edge_attr=[4246, 1024]),
     Data(x=[1536, 1024], edge_index=[2, 8149], edge_attr=[8149, 1024]),
     Data(x=[1981, 1024], edge_index=[2, 6006], edge_attr=[6006, 1024]),
     Data(x=[1119, 1024], edge_index=[2, 4501], edge_attr=[4501, 1024]),
     Data(x=[1395, 1024], edge_index=[2, 7217], edge_attr=[7217, 1024]),
     Data(x=[983, 1024], edge_index=[2, 2642], edge_attr=[2642, 1024]),
     Data(x=[1634, 1024], edge_index=[2, 3905], edge_attr=[3905, 1024]),
     Data(x=[1182, 1024], edge_index=[2, 3135], edge_attr=[3135, 1024]),
     Data(x=[703, 1024], edge_index=[2, 1575], edge_attr=[1575, 1024]),
     Data(x=[194, 1024], edge_index=[2, 428], edge_attr=[428, 1024]),
     Data(x=[876, 1024], edge_index=[2, 4971], edge_attr=[4971, 1024]),
     Data(x=[1964, 1024], edge_index=[2, 7721], edge_attr=[7721, 1024]),
     Data(x=[1956, 1024], edge_index=[2, 5400], edge_attr=[5400, 1024]),
     Data(x=[1918, 1024], edge_index=[2, 6171], edge_attr=[6171, 1024]),
     Data(x=[1351, 1024], edge_index=[2, 3741], edge_attr=[3741, 1024]),
     Data(x=[475, 1024], edge_index=[2, 1488], edge_attr=[1488, 1024]),
     Data(x=[1990, 1024], edge_index=[2, 5011], edge_attr=[5011, 1024]),
     Data(x=[509, 1024], edge_index=[2, 986], edge_attr=[986, 1024]),
     Data(x=[943, 1024], edge_index=[2, 2569], edge_attr=[2569, 1024]),
     Data(x=[739, 1024], edge_index=[2, 2404], edge_attr=[2404, 1024]),
     Data(x=[1674, 1024], edge_index=[2, 8595], edge_attr=[8595, 1024]),
     Data(x=[1998, 1024], edge_index=[2, 5444], edge_attr=[5444, 1024]),
     Data(x=[1223, 1024], edge_index=[2, 5361], edge_attr=[5361, 1024]),
     Data(x=[428, 1024], edge_index=[2, 1377], edge_attr=[1377, 1024]),
     Data(x=[1767, 1024], edge_index=[2, 4428], edge_attr=[4428, 1024]),
     Data(x=[404, 1024], edge_index=[2, 734], edge_attr=[734, 1024]),
     Data(x=[1416, 1024], edge_index=[2, 4094], edge_attr=[4094, 1024]),
     Data(x=[1658, 1024], edge_index=[2, 6257], edge_attr=[6257, 1024]),
     Data(x=[1907, 1024], edge_index=[2, 7995], edge_attr=[7995, 1024]),
     Data(x=[1992, 1024], edge_index=[2, 4590], edge_attr=[4590, 1024]),
     Data(x=[645, 1024], edge_index=[2, 1666], edge_attr=[1666, 1024]),
     Data(x=[1867, 1024], edge_index=[2, 4828], edge_attr=[4828, 1024]),
     Data(x=[1998, 1024], edge_index=[2, 5556], edge_attr=[5556, 1024]),
     Data(x=[1026, 1024], edge_index=[2, 3280], edge_attr=[3280, 1024]),
     Data(x=[1956, 1024], edge_index=[2, 7203], edge_attr=[7203, 1024]),
     Data(x=[1986, 1024], edge_index=[2, 6926], edge_attr=[6926, 1024]),
     Data(x=[836, 1024], edge_index=[2, 1527], edge_attr=[1527, 1024]),
     Data(x=[1367, 1024], edge_index=[2, 3654], edge_attr=[3654, 1024]),
     Data(x=[1695, 1024], edge_index=[2, 5494], edge_attr=[5494, 1024]),
     Data(x=[371, 1024], edge_index=[2, 722], edge_attr=[722, 1024]),
     Data(x=[1986, 1024], edge_index=[2, 6049], edge_attr=[6049, 1024]),
     Data(x=[815, 1024], edge_index=[2, 2322], edge_attr=[2322, 1024]),
     Data(x=[1026, 1024], edge_index=[2, 3285], edge_attr=[3285, 1024]),
     Data(x=[1233, 1024], edge_index=[2, 3088], edge_attr=[3088, 1024]),
     Data(x=[290, 1024], edge_index=[2, 577], edge_attr=[577, 1024]),
     Data(x=[1358, 1024], edge_index=[2, 4891], edge_attr=[4891, 1024]),
     Data(x=[1946, 1024], edge_index=[2, 6642], edge_attr=[6642, 1024]),
     Data(x=[406, 1024], edge_index=[2, 1000], edge_attr=[1000, 1024]),
     Data(x=[1973, 1024], edge_index=[2, 5091], edge_attr=[5091, 1024]),
     Data(x=[1124, 1024], edge_index=[2, 4301], edge_attr=[4301, 1024]),
     Data(x=[1530, 1024], edge_index=[2, 4502], edge_attr=[4502, 1024]),
     Data(x=[1020, 1024], edge_index=[2, 2425], edge_attr=[2425, 1024]),
     Data(x=[1410, 1024], edge_index=[2, 8048], edge_attr=[8048, 1024]),
     Data(x=[1998, 1024], edge_index=[2, 5556], edge_attr=[5556, 1024]),
     Data(x=[1996, 1024], edge_index=[2, 4360], edge_attr=[4360, 1024]),
     Data(x=[1867, 1024], edge_index=[2, 4828], edge_attr=[4828, 1024]),
     Data(x=[1866, 1024], edge_index=[2, 5171], edge_attr=[5171, 1024]),
     Data(x=[293, 1024], edge_index=[2, 422], edge_attr=[422, 1024])]



We expect the two results to be functionally identical, with the
differences being due to floating point jitter.

.. code:: ipython3

    def results_are_close_enough(ground_truth: Data, new_method: Data, thresh=.8):
        def _sorted_tensors_are_close(tensor1, tensor2):
            return torch.all(torch.isclose(tensor1.sort(dim=0)[0], tensor2.sort(dim=0)[0]).float().mean(axis=1) > thresh)
        def _graphs_are_same(tensor1, tensor2):
            return nx.weisfeiler_lehman_graph_hash(nx.Graph(tensor1.T)) == nx.weisfeiler_lehman_graph_hash(nx.Graph(tensor2.T))
        return _sorted_tensors_are_close(ground_truth.x, new_method.x) \
            and _sorted_tensors_are_close(ground_truth.edge_attr, new_method.edge_attr) \
            and _graphs_are_same(ground_truth.edge_index, new_method.edge_index)

.. code:: ipython3

    all_results_match = True
    for old_graph, new_graph in tqdm.tqdm(zip(dataset_graphs_embedded, dataset_graphs_embedded_largegraphindexer), total=num_questions):
        all_results_match &= results_are_close_enough(old_graph, new_graph)
    all_results_match


.. parsed-literal::

    100%|██████████| 100/100 [00:25<00:00,  4.00it/s]




.. parsed-literal::

    True



When scaled up to the entire dataset, we see a 2x speedup with indexing
this way.

WebQSPDataset is a question-by-question implementation.

UpdatedQSPDataset is a LargeGraphIndexer implementation.

These were computed on an RTX 4090 with 24GB of memory. Your milage may
vary.

Example 2: Building a new Dataset from Questions and an already-existing Knowledge Graph
----------------------------------------------------------------------------------------

Motivation
~~~~~~~~~~

One potential application of knowledge graph structural encodings is
capturing the relationships between different entities that are multiple
hops apart. This can be challenging for an LLM to recognize from
prepended graph information. Here’s a motivating example (credit to
@Rishi Puri):

.. code:: ipython3

    from IPython.display import SVG

.. code:: ipython3

    SVG(filename='./media/multihop_example.svg')




.. image:: 0_1_Encoding_from_Scratch_files/0_1_Encoding_from_Scratch_6_0.svg



In this example, the question can only be answered by reasoning about
the relationships between the entities in the knowledge graph.

Building a Multi-Hop QA Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To start, we need to download the raw data of a knowledge graph. In this
case, we use WikiData5M (`Wang et
al <https://paperswithcode.com/paper/kepler-a-unified-model-for-knowledge>`__).
Here we download the raw triplets and their entity codes. Information
about this dataset can be found
`here <https://deepgraphlearning.github.io/project/wikidata5m>`__.

The following download contains the ID to plaintext mapping for all the
entities and relations in the knowledge graph:

.. code:: ipython3

    !wget -O "https://www.dropbox.com/s/lnbhc8yuhit4wm5/wikidata5m_alias.tar.gz"

.. code:: ipython3

    !tar -xvf "wikidata5m_alias.tar.gz"

.. code:: ipython3

    with open('wikidata5m_entity.txt') as f:
        print(f.readline())

.. code:: ipython3

    with open('wikidata5m_relation.txt') as f:
        print(f.readline())

And then this download contains the raw triplets:

.. code:: ipython3

    !wget -O "https://www.dropbox.com/s/563omb11cxaqr83/wikidata5m_all_triplet.txt.gz"

.. code:: ipython3

    !gzip -d "wikidata5m_all_triplet.txt.gz" -f

.. code:: ipython3

    with open('wikidata5m_all_triplet.txt') as f:
        print(f.readline())

To start, we are going to preprocess the knowledge graph to substitute
each of the entity/relation codes with their plaintext aliases. This
makes it easier to use a pre-trained textual encoding model to create
triplet embeddings, as such a model likely won’t understand how to
properly embed the entity codes.

.. code:: ipython3

    import pandas as pd
    import tqdm
    import json

.. code:: ipython3

    # Substitute entity codes with their aliases
    # Picking the first alias for each entity (rather arbitrarily)
    alias_map = {}
    rel_alias_map = {}
    for line in open('wikidata5m_entity.txt'):
        parts = line.strip().split('\t')
        entity_id = parts[0]
        aliases = parts[1:]
        alias_map[entity_id] = aliases[0]
    for line in open('wikidata5m_relation.txt'):
        parts = line.strip().split('\t')
        relation_id = parts[0]
        relation_name = parts[1]
        rel_alias_map[relation_id] = relation_name

.. code:: ipython3

    full_graph = []
    missing_total = 0
    total = 0
    for line in tqdm.tqdm(open('wikidata5m_all_triplet.txt')):
        src, rel, dst = line.strip().split('\t')
        if src not in alias_map:
            missing_total += 1
        if dst not in alias_map:
            missing_total += 1
        if rel not in rel_alias_map:
            missing_total += 1
        total += 3
        full_graph.append([alias_map.get(src, src), rel_alias_map.get(rel, rel), alias_map.get(dst, dst)])
    print(f"Missing aliases: {missing_total}/{total}")

.. code:: ipython3

    full_graph[:10]

Now ``full_graph`` represents the knowledge graph triplets in
understandable plaintext.

Next, we need a set of multi-hop questions that the Knowledge Graph will
provide us with context for. We utilize a subset of
`HotPotQA <https://hotpotqa.github.io/>`__ (`Yang et.
al. <https://arxiv.org/pdf/1809.09600>`__) called
`2WikiMultiHopQA <https://github.com/Alab-NII/2wikimultihop>`__ (`Ho et.
al. <https://aclanthology.org/2020.coling-main.580.pdf>`__), which
includes a subgraph of entities that serve as the ground truth
justification for answering each multi-hop question:

.. code:: ipython3

    !wget -O "https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip"

.. code:: ipython3

    !unzip -o "data_ids_april7.zip"

.. code:: ipython3

    with open('train.json') as f:
        train_data = json.load(f)
    train_df = pd.DataFrame(train_data)
    train_df['split_type'] = 'train'
    
    with open('dev.json') as f:
        dev_data = json.load(f)
    dev_df = pd.DataFrame(dev_data)
    dev_df['split_type'] = 'dev'
    
    with open('test.json') as f:
        test_data = json.load(f)
    test_df = pd.DataFrame(test_data)
    test_df['split_type'] = 'test'
    
    df = pd.concat([train_df, dev_df, test_df])

.. code:: ipython3

    df.head()

.. code:: ipython3

    df['split_type'].value_counts()

.. code:: ipython3

    df['type'].value_counts()

Now we need to extract the subgraphs

.. code:: ipython3

    df['graph_size'] = df['evidences_id'].apply(lambda row: len(row))

.. code:: ipython3

    df['graph_size'].value_counts()

(Optional) We take only questions where the evidence graph is greater
than 0. (Note: this gets rid of the test set):

.. code:: ipython3

    # df = df[df['graph_size'] > 0]

.. code:: ipython3

    df['split_type'].value_counts()

.. code:: ipython3

    df.columns

.. code:: ipython3

    refined_df = df[['_id', 'question', 'answer', 'split_type', 'evidences_id', 'type', 'graph_size']]

.. code:: ipython3

    refined_df.head()

Checkpoint:

.. code:: ipython3

    refined_df.to_csv('wikimultihopqa_refined.csv', index=False)

Now we need to check that all the entities mentioned in the
question/answer set are also present in the Wikidata graph:

.. code:: ipython3

    relation_map = {}
    with open('wikidata5m_relation.txt') as f:
        for line in tqdm.tqdm(f):
            parts = line.strip().split('\t')
            for i in range(1, len(parts)):
                if parts[i] not in relation_map:
                    relation_map[parts[i]] = []
                relation_map[parts[i]].append(parts[0])

.. code:: ipython3

    # Manually check to see if all of these are valid in WikiData DB, even if they may not be answerable in WikiData5M
    for row in refined_df.itertuples():
        for trip in row.evidences_id:
            relation = trip[1]
            if relation not in relation_map:
                print(f'The following relation is not found: {relation}')
            elif len(relation_map[relation]) > 1:
                print(f'The following relation alias has a collision: {relation}: {relation_map[relation]}')

.. code:: ipython3

    entity_set = set()
    with open('wikidata5m_entity.txt') as f:
        for line in tqdm.tqdm(f):
            entity_set.add(line.strip().split('\t')[0])

.. code:: ipython3

    missing_entities = set()
    missing_entity_idx = set()
    for i, row in enumerate(refined_df.itertuples()):
        for trip in row.evidences_id:
            if len(trip) != 3:
                print(trip)
            entities = trip[0], trip[2]
            for entity in entities:
                if entity not in entity_set:
                    print(f'The following entity was not found in the KG: {entity}')
                    missing_entities.add(entity)
                    missing_entity_idx.add(i)

Right now, we drop the missing entity entries. Additional preprocessing
can be done here to resolve the entity/relation collisions, but that is
out of the scope for this notebook.

.. code:: ipython3

    len(missing_entity_idx)

.. code:: ipython3

    refined_df.shape

.. code:: ipython3

    # missing relations are ok, but missing entities cannot be mapped to plaintext, so they should be dropped.
    refined_df.reset_index(inplace=True, drop=True)
    refined_df

.. code:: ipython3

    cleaned_df = refined_df.drop(missing_entity_idx)
    cleaned_df['split_type'].value_counts()

Now we save the resulting graph and questions/answers dataset:

.. code:: ipython3

    cleaned_df.to_csv('wikimultihopqa_cleaned.csv', index=False)

.. code:: ipython3

    import torch

.. code:: ipython3

    torch.save(full_graph, 'wikimultihopqa_full_graph.pt')

Question: How do we extract a contextual subgraph for a given query?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The chosen retrieval algorithm is a critical component in the pipeline
for affecting RAG performance. In the next section (1), we will
demonstrate a naive method of retrieval for a large knowledge graph, and
how to apply it to this dataset along with WebQSP.

Preparing a Textualized Graph for LLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For now however, we need to prepare the graph data to be used as a
plaintext prefix to the LLM. In order to do this, we want to prompt the
LLM to use the unique nodes, and unique edge triplets of a given
subgraph. In order to do this, we prepare a unique indexed node df and
edge df for the knowledge graph now. This process occurs trivially with
the LargeGraphIndexer:

.. code:: ipython3

    from torch_geometric.data import LargeGraphIndexer

.. code:: ipython3

    indexer = LargeGraphIndexer.from_triplets(full_graph)

.. code:: ipython3

    # Node DF
    textual_nodes = pd.DataFrame.from_dict(
        {"node_attr": indexer.get_node_features()})
    textual_nodes["node_id"] = textual_nodes.index
    textual_nodes = textual_nodes[["node_id", "node_attr"]]

.. code:: ipython3

    textual_nodes.head()

Notice how LargeGraphIndexer ensures that there are no duplicate
indices:

.. code:: ipython3

    textual_nodes['node_attr'].unique().shape[0]/textual_nodes.shape[0]

.. code:: ipython3

    # Edge DF
    textual_edges = pd.DataFrame(indexer.get_edge_features(),
                                    columns=["src", "edge_attr", "dst"])
    textual_edges["src"] = [
        indexer._nodes[h] for h in textual_edges["src"]
    ]
    textual_edges["dst"] = [
        indexer._nodes[h] for h in textual_edges["dst"]
    ]

Note: The edge table refers to each node by its index in the node table.
We will see how this gets utilized later when indexing a subgraph.

.. code:: ipython3

    textual_edges.head()

Now we can save the result

.. code:: ipython3

    textual_nodes.to_csv('wikimultihopqa_textual_nodes.csv', index=False)
    textual_edges.to_csv('wikimultihopqa_textual_edges.csv', index=False)

Now were done! This knowledge graph and dataset will get used later on
in Section 1.

.. code:: ipython3

    # TODO: Refactor everything below this point into its own notebook

Generating Subgraphs
====================

.. code:: ipython3

    from profiling_utils import create_remote_backend_from_triplets
    from rag_feature_store import SentenceTransformerFeatureStore
    from rag_graph_store import NeighborSamplingRAGGraphStore
    from torch_geometric.loader import RAGQueryLoader
    from torch_geometric.nn.nlp import SentenceTransformer
    from torch_geometric.datasets.updated_web_qsp_dataset import preprocess_triplet, retrieval_via_pcst
    from torch_geometric.data import get_features_for_triplets_groups, Data

.. code:: ipython3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(model_name='sentence-transformers/all-roberta-large-v1').to(device)

.. code:: ipython3

    fs, gs = create_remote_backend_from_triplets(full_graph, model, model, NeighborSamplingRAGGraphStore, SentenceTransformerFeatureStore, 'encode', 'encode', preprocess_triplet, 'wikidata_graph', node_method_kwargs={"batch_size": 256}).load()

.. code:: ipython3

    import torch
    import pandas as pd

.. code:: ipython3

    graphs = torch.load('subg_results.pt')
    graph_df = pd.read_csv("wikimultihopqa_cleaned.csv")

.. code:: ipython3

    graph_df['is_train'].value_counts()

.. code:: ipython3

    graph_df.head()

.. code:: ipython3

    graph_df['is_train'].value_counts()


Retrieval Algorithms and Scaling Retrieval
==========================================

Motivation
----------

When building a RAG Pipeline for inference, the retrieval component is
important for the following reasons: 1. A given algorithm for retrieving
subgraph context can have a marked effect on the hallucination rate of
the responses in the model 2. A given retrieval algorithm needs to be
able to scale to larger graphs of millions of nodes and edges in order
to be practical for production.

In this notebook, we will explore how to construct a RAG retrieval
algorithm from a given subgraph, and conduct some experiments to
evaluate its runtime performance.

We want to do so in-line with Pytorch Geometric’s in-house framework for
remote backends:

.. code:: ipython3

    from IPython.display import Image, SVG
    Image(filename='../../../docs/source/_figures/remote_2.png')




.. image:: 1_Retrieval_files/1_Retrieval_5_0.png



As seen here, the GraphStore is used to store the neighbor relations
between the nodes of the graph, whereas the FeatureStore is used to
store the node and edge features in the graph.

Let’s start by loading in a knowledge graph dataset for the sake of our
experiment:

.. code:: ipython3

    from torch_geometric.data import LargeGraphIndexer
    from torch_geometric.datasets import UpdatedWebQSPDataset
    from itertools import chain


.. parsed-literal::

    /home/zaristei/.local/lib/python3.10/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm


.. code:: ipython3

    # Limiting to 10 questions for the sake of compute, but can be increased if necessary
    ds = UpdatedWebQSPDataset(root='demo', limit=10)

Let’s set up our set of questions and graph triplets:

.. code:: ipython3

    questions = ds.raw_dataset['question']
    questions




.. parsed-literal::

    ['what is the name of justin bieber brother',
     'what character did natalie portman play in star wars',
     'what country is the grand bahama island in',
     'what kind of money to take to bahamas',
     'what character did john noble play in lord of the rings',
     'who does joakim noah play for',
     'where are the nfl redskins from',
     'where did saki live',
     'who did draco malloy end up marrying',
     'which countries border the us']



.. code:: ipython3

    ds.raw_dataset[:10]['graph'][0][:10]




.. parsed-literal::

    [['P!nk', 'freebase.valuenotation.is_reviewed', 'Gender'],
     ['1Club.FM: Power', 'broadcast.content.artist', 'P!nk'],
     ['Somebody to Love', 'music.recording.contributions', 'm.0rqp4h0'],
     ['Rudolph Valentino', 'freebase.valuenotation.is_reviewed', 'Place of birth'],
     ['Ice Cube', 'broadcast.artist.content', '.977 The Hits Channel'],
     ['Colbie Caillat', 'broadcast.artist.content', 'Hot Wired Radio'],
     ['Stephen Melton', 'people.person.nationality', 'United States of America'],
     ['Record producer',
      'music.performance_role.regular_performances',
      'm.012m1vf1'],
     ['Justin Bieber', 'award.award_winner.awards_won', 'm.0yrkc0l'],
     ['1.FM Top 40', 'broadcast.content.artist', 'Geri Halliwell']]



.. code:: ipython3

    all_triplets = chain.from_iterable((row['graph'] for row in ds.raw_dataset))

With these questions and triplets, we want to: 1. Consolidate all the
relations in these triplets into a Knowledge Graph 2. Create a
FeatureStore that encodes all the nodes and edges in the knowledge graph
3. Create a GraphStore that encodes all the edge indices in the
knowledge graph

.. code:: ipython3

    import torch
    from torch_geometric.nn.nlp import SentenceTransformer
    from torch_geometric.datasets.updated_web_qsp_dataset import preprocess_triplet

.. code:: ipython3

    import sys
    sys.path.append('..')

In order to create a remote backend, we need to define a FeatureStore
and GraphStore locally, as well as a method for initializing its state
from triplets:

.. code:: ipython3

    from profiling_utils import create_remote_backend_from_triplets, RemoteGraphBackendLoader
    
    # We define this GraphStore to sample the neighbors of a node locally.
    # Ideally for a real remote backend, this interface would be replaced with an API to a Graph DB, such as Neo4j.
    from rag_graph_store import NeighborSamplingRAGGraphStore
    
    # We define this FeatureStore to encode the nodes and edges locally, and perform appoximate KNN when indexing.
    # Ideally for a real remote backend, this interface would be replaced with an API to a vector DB, such as Pinecone.
    from rag_feature_store import SentenceTransformerFeatureStore

.. code:: ipython3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(model_name="sentence-transformers/all-roberta-large-v1").to(device)
    
    backend_loader: RemoteGraphBackendLoader = create_remote_backend_from_triplets(
        triplets=all_triplets, # All the triplets to insert into the backend
        node_embedding_model=model, # Embedding model to process triplets with
        node_method_to_call="encode", # This method will encode the nodes/edges with 'model.encode' in this case.
        path="backend", # Save path
        pre_transform=preprocess_triplet, # Preprocessing function to apply to triplets before invoking embedding model.
        node_method_kwargs={"batch_size": 256}, # Keyword arguments to pass to the node_method_to_call.
        graph_db=NeighborSamplingRAGGraphStore, # Graph Store to use
        feature_db=SentenceTransformerFeatureStore # Feature Store to use
        ) 
    # This loader saves a copy of the processed data locally to be transformed into a graphstore and featurestore when load() is called.
    feature_store, graph_store = backend_loader.load()

Now that we have initialized our remote backends, we can now retrieve
from them using a Loader to query the backends, as shown in this
diagram:

.. code:: ipython3

    Image(filename='../../../docs/source/_figures/remote_3.png')




.. image:: 1_Retrieval_files/1_Retrieval_21_0.png



.. code:: ipython3

    from torch_geometric.loader import RAGQueryLoader

.. code:: ipython3

    query_loader = RAGQueryLoader(
        data=(feature_store, graph_store), # Remote Rag Graph Store and Feature Store
        # Arguments to pass into the seed node/edge retrieval methods for the FeatureStore.
        # In this case, it's k for the KNN on the nodes and edges.
        seed_nodes_kwargs={"k_nodes": 10}, seed_edges_kwargs={"k_edges": 10}, 
        # Arguments to pass into the GraphStore's Neighbor sampling method.
        # In this case, the GraphStore implements a NeighborLoader, so it takes the same arguments.
        sampler_kwargs={"num_neighbors": [40]*3},
        # Arguments to pass into the FeatureStore's feature loading method.
        loader_kwargs={},
        # An optional local transform that can be applied on the returned subgraph.
        local_filter=None,
        )

To make better sense of this loader’s arguments, let’s take a closer
look at the retrieval process for a remote backend:

.. code:: ipython3

    SVG(filename="media/remote_backend.svg")




.. image:: 1_Retrieval_files/1_Retrieval_25_0.svg



As we see here, there are 3 important steps to any remote backend
procedure for graphs: 1. Retrieve the seed nodes and edges to begin our
retrieval process from. 2. Traverse the graph neighborhood of the seed
nodes/edges to gather local context. 3. Fetch the features associated
with the subgraphs obtained from the traversal.

We can see that our Query Loader construction allows us to specify
unique hyperparameters for each unique step in this retrieval.

Now we can submit our queries to the remote backend to retrieve our
subgraphs:

.. code:: ipython3

    import tqdm

.. code:: ipython3

    sub_graphs = []
    for q in tqdm.tqdm(questions):
        sub_graphs.append(query_loader.query(q))


.. parsed-literal::

      0%|          | 0/10 [00:00<?, ?it/s]

.. parsed-literal::

    torch.Size([11466, 1024])
    torch.Size([38145, 1024])


.. parsed-literal::

    100%|██████████| 10/10 [00:07<00:00,  1.28it/s]


.. code:: ipython3

    sub_graphs[0]




.. parsed-literal::

    Data(x=[2251, 1024], edge_index=[2, 7806], edge_attr=[7806, 1024], node_idx=[2251], edge_idx=[7806])



These subgraphs are now retrieved using a different retrieval method
when compared to the original WebQSP dataset. Can we compare the
properties of this method to the original WebQSPDataset’s retrieval
method? Let’s compare some basics properties of the subgraphs:

.. code:: ipython3

    from torch_geometric.data import Data

.. code:: ipython3

    def _eidx_helper(subg: Data, ground_truth: Data):
        subg_eidx, gt_eidx = subg.edge_idx, ground_truth.edge_idx
        if isinstance(subg_eidx, torch.Tensor):
            subg_eidx = subg_eidx.tolist()
        if isinstance(gt_eidx, torch.Tensor):
            gt_eidx = gt_eidx.tolist()
        subg_e = set(subg_eidx)
        gt_e = set(gt_eidx)
        return subg_e, gt_e
    def check_retrieval_accuracy(subg: Data, ground_truth: Data, num_edges: int):
        subg_e, gt_e = _eidx_helper(subg, ground_truth)
        total_e = set(range(num_edges))
        tp = len(subg_e & gt_e)
        tn = len(total_e-(subg_e | gt_e))
        return (tp+tn)/num_edges
    def check_retrieval_precision(subg: Data, ground_truth: Data):
        subg_e, gt_e = _eidx_helper(subg, ground_truth)
        return len(subg_e & gt_e) / len(subg_e)
    def check_retrieval_recall(subg: Data, ground_truth: Data):
        subg_e, gt_e = _eidx_helper(subg, ground_truth)
        return len(subg_e & gt_e) / len(gt_e)

.. code:: ipython3

    from torch_geometric.data import get_features_for_triplets_groups

.. code:: ipython3

    ground_truth_graphs = get_features_for_triplets_groups(ds.indexer, (d['graph'] for d in ds.raw_dataset), pre_transform=preprocess_triplet)
    num_edges = len(ds.indexer._edges)

.. code:: ipython3

    for subg, ground_truth in tqdm.tqdm(zip((query_loader.query(q) for q in questions), ground_truth_graphs)):
        print(f"Size: {len(subg.x)}, Ground Truth Size: {len(ground_truth.x)}, Accuracy: {check_retrieval_accuracy(subg, ground_truth, num_edges)}, Precision: {check_retrieval_precision(subg, ground_truth)}, Recall: {check_retrieval_recall(subg, ground_truth)}")


.. parsed-literal::

    10it [00:00, 60.20it/s]
    1it [00:00,  1.18it/s]

.. parsed-literal::

    Size: 2193, Ground Truth Size: 1709, Accuracy: 0.6636780705203827, Precision: 0.22923807012918535, Recall: 0.1994037381034285


.. parsed-literal::

    2it [00:01,  1.41it/s]

.. parsed-literal::

    Size: 2682, Ground Truth Size: 1251, Accuracy: 0.7158736400576746, Precision: 0.10843513670738801, Recall: 0.22692963233503774


.. parsed-literal::

    3it [00:02,  1.51it/s]

.. parsed-literal::

    Size: 2087, Ground Truth Size: 1285, Accuracy: 0.7979813868134749, Precision: 0.0547879177377892, Recall: 0.15757855822550831


.. parsed-literal::

    4it [00:02,  1.56it/s]

.. parsed-literal::

    Size: 2975, Ground Truth Size: 1988, Accuracy: 0.6956088609254162, Precision: 0.14820555621795636, Recall: 0.21768826619964973


.. parsed-literal::

    5it [00:03,  1.59it/s]

.. parsed-literal::

    Size: 2594, Ground Truth Size: 633, Accuracy: 0.78849128326124, Precision: 0.04202616198163095, Recall: 0.2032301480484522


.. parsed-literal::

    6it [00:03,  1.61it/s]

.. parsed-literal::

    Size: 2462, Ground Truth Size: 1044, Accuracy: 0.7703499803381832, Precision: 0.07646643109540636, Recall: 0.19551861221539574


.. parsed-literal::

    7it [00:04,  1.62it/s]

.. parsed-literal::

    Size: 2011, Ground Truth Size: 1382, Accuracy: 0.7871804954777821, Precision: 0.10117783355860205, Recall: 0.13142713819914723


.. parsed-literal::

    8it [00:05,  1.63it/s]

.. parsed-literal::

    Size: 2011, Ground Truth Size: 1052, Accuracy: 0.802831301612269, Precision: 0.06452691407556001, Recall: 0.16702726092600606


.. parsed-literal::

    9it [00:05,  1.64it/s]

.. parsed-literal::

    Size: 2892, Ground Truth Size: 1012, Accuracy: 0.7276182985974571, Precision: 0.10108615156751419, Recall: 0.20860927152317882


.. parsed-literal::

    10it [00:06,  1.58it/s]

.. parsed-literal::

    Size: 1817, Ground Truth Size: 1978, Accuracy: 0.7530475815965395, Precision: 0.1677807486631016, Recall: 0.11696178937558248


.. parsed-literal::

    


Note that, since we’re only comparing the results of 10 graphs here,
this retrieval algorithm is not taking into account the full corpus of
nodes in the dataset. If you want to see a full example, look at
``rag_generate.py``, or ``rag_generate_multihop.py`` These examples
generate datasets for the entirety of the WebQSP dataset, or the
WikiData Multihop datasets that are discussed in Section 0.

Evaluating Runtime Performance
------------------------------

Pytorch Geometric provides multiple methods for evalutaing runtime
performance. In this notebook, we utilize NVTX to profile the different
components of our RAG Query Loader.

The method ``nvtxit`` allows for profiling the utilization and timings
of any methods that get wrapped by it in a Python script.

To see an example of this, check out
``nvtx_examples/nvtx_rag_backend_example.py``.

This script mirrors this notebook’s functionality, but notably, it
includes the following code snippet:

.. code:: python

   # Patch FeatureStore and GraphStore

   SentenceTransformerFeatureStore.retrieve_seed_nodes = nvtxit()(SentenceTransformerFeatureStore.retrieve_seed_nodes)
   SentenceTransformerFeatureStore.retrieve_seed_edges = nvtxit()(SentenceTransformerFeatureStore.retrieve_seed_edges)
   SentenceTransformerFeatureStore.load_subgraph = nvtxit()(SentenceTransformerFeatureStore.load_subgraph)
   NeighborSamplingRAGGraphStore.sample_subgraph = nvtxit()(NeighborSamplingRAGGraphStore.sample_subgraph)
   rag_loader.RAGQueryLoader.query = nvtxit()(rag_loader.RAGQueryLoader.query)

Importantly, this snippet wraps the methods of FeatureStore, GraphStore,
and the Query method from QueryLoader so that it will be recognized as a
unique frame in NVTX.

This can be executed by the included shell script ``nvtx_run.sh``:

.. code:: bash

   ...

   # Get the base name of the Python file
   python_file=$(basename "$1")

   # Run nsys profile on the Python file
   nsys profile -c cudaProfilerApi --capture-range-end repeat -t cuda,nvtx,osrt,cudnn,cublas --cuda-memory-usage true --cudabacktrace all --force-overwrite true --output=profile_${python_file%.py} python "$1"

   echo "Profile data saved as profile_${python_file%.py}.nsys-rep"

The generated resulting ``.nsys-rep`` file can be visualized using tools
like Nsight Systems or Nsight Compute, that can show the relative
timings of the FeatureStore, GraphStore, and QueryLoader methods.
