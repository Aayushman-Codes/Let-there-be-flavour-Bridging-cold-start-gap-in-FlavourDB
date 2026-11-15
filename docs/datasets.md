# SavorLink Dataset Reference & Provenance

This document provides a comprehensive inventory, schema specification, curation methodology, and official download links for all public datasets and ontological resources utilized in this project.

---

## Deliverables & Dataset Provenance Statement

This project relies entirely on **publicly available, peer-reviewed scientific datasets and open-access biochemical databases**. All datasets used in this project are listed below along with their official repositories, primary academic literature citations, and schema descriptions.

Precomputed chemical fingerprints, semantic embeddings, and similarity matrices are cached inside `data/cache/` to ensure deterministic, zero-dependency, and rapid reproduction on commodity hardware without requiring external API tokens or heavy chemistry dependencies (e.g., RDKit). All caches can be regenerated from scratch using `python -m src.data.build_all_modalities`.

---

## 1. FlavorGraph

### Overview
FlavorGraph is a large-scale, heterogeneous food-chemical bipartite network that unifies culinary ingredients with their constituent volatile and non-volatile chemical compounds, alongside co-occurrence statistics mined from over one million culinary recipes.

- **Primary Literature:**
  Park, D., Kim, K., Kim, S. et al. *"FlavorGraph: a large-scale food-chemical graph for generating food representations and recommending food pairings"*, *Scientific Reports* 11, 931 (2021).
  - DOI: [10.1038/s41598-020-79425-8](https://doi.org/10.1038/s41598-020-79425-8)
  - URL: [https://www.nature.com/articles/s41598-020-79425-8](https://www.nature.com/articles/s41598-020-79425-8)
- **Official Source Code & Dataset Repository:**
  - GitHub: [https://github.com/lamypark/FlavorGraph](https://github.com/lamypark/FlavorGraph)
  - License: Apache License 2.0

### Local Files & Schemas

#### Nodes (`data/raw/nodes_191120.csv`)
- **Total Entities:** 8,214 nodes (6,653 culinary ingredients + 1,561 chemical compounds)
- **Hub Ingredients:** 416 ingredients possess chemical compound associations and form the connected hub core.
- **Columns:**
  - `node_id`: Integer primary key identifier ($0 \dots 8213$)
  - `name`: English entity descriptor (e.g., `garlic`, `allicin`, `cheddar_cheese`)
  - `id`: PubChem Compound Identifier (CID) for chemical nodes; empty for ingredient nodes
  - `node_type`: Categorical indicator (`ingredient` or `compound`)
  - `is_hub`: Categorical indicator (`hub` for hub ingredients, `food` or `drug` for compound subtypes)

#### Edges (`data/raw/edges_191120.csv`)
- **Total Edges:** 40,654 directed/undirected relationships
- **Columns:**
  - `id_1`: Source `node_id`
  - `id_2`: Destination `node_id`
  - `score`: Normalized Pointwise Mutual Information (NPMI) for recipe pairings ($[-1, 1]$); empty for ingredient-compound links
  - `edge_type`: Categorical edge relationship:
    - `ingr-ingr`: Culinary co-occurrence derived from Recipe1M+ recipes
    - `ingr-fcomp`: Ingredient contains food-derived flavor compound
    - `ingr-dcomp`: Ingredient contains drug-like or bioactive compound

---

## 2. Recipe1M+

### Overview
Recipe1M+ is a massive structured collection of culinary recipes scraped from cooking websites, providing structured lists of ingredients, cooking instructions, and nutritional attributes. The `ingr-ingr` edges in FlavorGraph represent Normalized Pointwise Mutual Information (NPMI) computed over the co-occurrence frequencies in Recipe1M+.

- **Primary Literature:**
  - Marin, J., Biswas, A., Ofli, F., Hynes, N., Salvador, A., Aytar, Y., Weber, I., & Torralba, A. *"Recipe1M+: A Dataset for Structured Recipe Representation"*, *IEEE Transactions on Pattern Analysis and Machine Intelligence* (TPAMI), 2019.
    - DOI: [10.1109/TPAMI.2019.2927439](https://doi.org/10.1109/TPAMI.2019.2927439)
  - Salvador, A., Hynes, N., Aytar, Y., Marin, J., Ofli, F., Weber, I., & Torralba, A. *"Inverse Cooking: Recipe Generation from Food Images"*, *IEEE/CVF Conference on Computer Vision and Pattern Recognition* (CVPR), 2019.
- **Official Dataset Portal:**
  - URL: [http://pic2recipe.csail.mit.edu/](http://pic2recipe.csail.mit.edu/)

---

## 3. FlavorDB

### Overview
FlavorDB is a curated repository of flavor molecules, providing comprehensive catalogs of volatile chemicals, flavor profiles (e.g., fruity, nutty, floral, sulfurous), and their presence across natural food ingredients.

- **Primary Literature:**
  Garg, N., Sethupathy, A., Tuwani, R., Singh, R., Saha, S., Agarwal, N., & Bagler, G. *"FlavorDB: a resource of flavor molecules"*, *Nucleic Acids Research*, Volume 46, Issue D1, 4 January 2018, Pages D1210–D1216.
  - DOI: [10.1093/nar/gkx1094](https://doi.org/10.1093/nar/gkx1094)
  - URL: [https://academic.oup.com/nar/article/46/D1/D1210/4563459](https://academic.oup.com/nar/article/46/D1/D1210/4563459)
- **Official Database Portal:**
  - URL: [https://cosman.iitd.ac.in/flavordb/](https://cosman.iitd.ac.in/flavordb/)

---

## 4. PubChem

### Overview
PubChem is an open chemistry database maintained by the National Center for Biotechnology Information (NCBI) at the U.S. National Institutes of Health (NIH). SavorLink resolves PubChem CIDs from FlavorGraph nodes to canonical Simplified Molecular-Input Line-Entry System (SMILES) chemical representations.

- **Primary Literature:**
  Kim, S., Chen, J., Cheng, T. et al. *"PubChem 2023 update"*, *Nucleic Acids Research*, 51(D1), D1373–D1380 (2023).
  - DOI: [10.1093/nar/gkac956](https://doi.org/10.1093/nar/gkac956)
- **Official Database Portal & REST API:**
  - URL: [https://pubchem.ncbi.nlm.nih.gov/](https://pubchem.ncbi.nlm.nih.gov/)
  - REST PUG API: `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/<CID>/property/CanonicalSMILES/JSON`
- **Cached Resource:** `data/cache/pubchem_smiles.json` (contains verified SMILES for all 1,561 compounds)

---

## 5. FoodOn (Food Ontology)

### Overview
FoodOn is a consortium-driven, harmonized ontologic resource that represents agricultural, culinary, and nutritional concepts across food origins, taxonomies, processing stages, and consumption patterns.

- **Primary Literature:**
  Dooley, D.M., Griffiths, E.J., Gosal, G.S. et al. *"FoodOn: a harmonized ontologic resource in food, agriculture and nutrition"*, *npj Science of Food* 2, 23 (2018).
  - DOI: [10.1038/s41538-018-0032-6](https://doi.org/10.1038/s41538-018-0032-6)
- **Official Ontology Portal & GitHub:**
  - URL: [https://foodon.org/](https://foodon.org/)
  - Repository: [https://github.com/FoodOntology/foodon](https://github.com/FoodOntology/foodon)
- **Implementation Note:**
  In SavorLink, food taxonomy is structured into a 3-level food hierarchy (`origin`, `group`, `subgroup`) implemented in `src/data/taxonomic_lookup.py` (`TAXONOMIC_RULES`) and serialized to `data/cache/foodon_taxonomy.json`.
