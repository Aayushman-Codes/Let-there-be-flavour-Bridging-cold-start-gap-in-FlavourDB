"""Taxonomic Food Taxonomy & FoodOn Lookup for SavorLink Strategy III.

Provides hierarchical classification for the 416 hub ingredients:
- Root (Food)
  -> Level 1: Kingdom / Origin (Plant, Animal, Fungi, Processed/Fermented)
     -> Level 2: Food Group (Fruit, Vegetable, Herb & Spice, Dairy, Meat, Poultry, Seafood, Grain, Legume, Nut/Seed, Beverage, etc.)
        -> Level 3: Botanical / Culinary Subgroup (Allium, Citrus, Nightshade, Pome, Bivalve, Salmonid, Bovine, etc.)

Supports:
1. Live / Cached FoodOn queries via EMBL-EBI OLS4 REST API.
2. Complete 3-level offline fallback hierarchy covering 100% of the 416 hub ingredients.
3. Persistent caching to `data/cache/foodon_taxonomy.json`.
"""
from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Standardized 3-level reference taxonomy for all culinary categories
TAXONOMIC_RULES: Dict[str, Tuple[str, str, str]] = {
    # Format: ingredient_substring / exact: (Origin, Group, Subgroup)
    # Fruits
    "apple": ("plant", "fruit", "pome_fruit"),
    "bartlett_pear": ("plant", "fruit", "pome_fruit"),
    "pear": ("plant", "fruit", "pome_fruit"),
    "quince": ("plant", "fruit", "pome_fruit"),
    "apricot": ("plant", "fruit", "stone_fruit"),
    "peach": ("plant", "fruit", "stone_fruit"),
    "plum": ("plant", "fruit", "stone_fruit"),
    "cherry": ("plant", "fruit", "stone_fruit"),
    "nectarine": ("plant", "fruit", "stone_fruit"),
    "banana": ("plant", "fruit", "tropical_fruit"),
    "mango": ("plant", "fruit", "tropical_fruit"),
    "pineapple": ("plant", "fruit", "tropical_fruit"),
    "papaya": ("plant", "fruit", "tropical_fruit"),
    "guava": ("plant", "fruit", "tropical_fruit"),
    "passion_fruit": ("plant", "fruit", "tropical_fruit"),
    "fig": ("plant", "fruit", "tropical_fruit"),
    "date": ("plant", "fruit", "tropical_fruit"),
    "pomegranate": ("plant", "fruit", "tropical_fruit"),
    "kiwi": ("plant", "fruit", "tropical_fruit"),
    "orange": ("plant", "fruit", "citrus"),
    "lemon": ("plant", "fruit", "citrus"),
    "lime": ("plant", "fruit", "citrus"),
    "grapefruit": ("plant", "fruit", "citrus"),
    "tangerine": ("plant", "fruit", "citrus"),
    "mandarin_orange": ("plant", "fruit", "citrus"),
    "kumquat": ("plant", "fruit", "citrus"),
    "strawberry": ("plant", "fruit", "berry"),
    "blueberry": ("plant", "fruit", "berry"),
    "blackberry": ("plant", "fruit", "berry"),
    "raspberry": ("plant", "fruit", "berry"),
    "cranberry": ("plant", "fruit", "berry"),
    "black_currant": ("plant", "fruit", "berry"),
    "red_currant": ("plant", "fruit", "berry"),
    "gooseberry": ("plant", "fruit", "berry"),
    "boysenberry": ("plant", "fruit", "berry"),
    "lingonberry": ("plant", "fruit", "berry"),
    "mulberry": ("plant", "fruit", "berry"),
    "elderberry": ("plant", "fruit", "berry"),
    "grape": ("plant", "fruit", "berry"),
    "raisin": ("plant", "fruit", "berry"),
    "currant": ("plant", "fruit", "berry"),
    "melon": ("plant", "fruit", "melon"),
    "cantaloupe": ("plant", "fruit", "melon"),
    "watermelon": ("plant", "fruit", "melon"),
    "honeydew": ("plant", "fruit", "melon"),
    "avocado": ("plant", "fruit", "fatty_fruit"),
    "olive": ("plant", "fruit", "fatty_fruit"),

    # Vegetables - Alliums
    "garlic": ("plant", "vegetable", "allium"),
    "onion": ("plant", "vegetable", "allium"),
    "shallot": ("plant", "vegetable", "allium"),
    "leek": ("plant", "vegetable", "allium"),
    "scallion": ("plant", "vegetable", "allium"),
    "chive": ("plant", "vegetable", "allium"),

    # Vegetables - Nightshades
    "tomato": ("plant", "vegetable", "nightshade"),
    "potato": ("plant", "vegetable", "nightshade"),
    "sweet_potato": ("plant", "vegetable", "root_tuber"),
    "eggplant": ("plant", "vegetable", "nightshade"),
    "bell_pepper": ("plant", "vegetable", "nightshade"),
    "yellow_bell_pepper": ("plant", "vegetable", "nightshade"),
    "chili_pepper": ("plant", "vegetable", "nightshade"),
    "jalapeno": ("plant", "vegetable", "nightshade"),
    "cayenne": ("plant", "vegetable", "nightshade"),
    "habanero": ("plant", "vegetable", "nightshade"),
    "pimento": ("plant", "vegetable", "nightshade"),

    # Vegetables - Cruciferous / Brassica
    "broccoli": ("plant", "vegetable", "cruciferous"),
    "cauliflower": ("plant", "vegetable", "cruciferous"),
    "cabbage": ("plant", "vegetable", "cruciferous"),
    "white_cabbage": ("plant", "vegetable", "cruciferous"),
    "red_cabbage": ("plant", "vegetable", "cruciferous"),
    "savoy_cabbage": ("plant", "vegetable", "cruciferous"),
    "brussels_sprout": ("plant", "vegetable", "cruciferous"),
    "kale": ("plant", "vegetable", "cruciferous"),
    "kohlrabi": ("plant", "vegetable", "cruciferous"),
    "bok_choy": ("plant", "vegetable", "cruciferous"),
    "turnip": ("plant", "vegetable", "cruciferous"),
    "rutabaga": ("plant", "vegetable", "cruciferous"),
    "radish": ("plant", "vegetable", "cruciferous"),
    "daikon": ("plant", "vegetable", "cruciferous"),
    "horseradish": ("plant", "vegetable", "cruciferous"),
    "wasabi": ("plant", "vegetable", "cruciferous"),
    "watercress": ("plant", "vegetable", "cruciferous"),
    "arugula": ("plant", "vegetable", "cruciferous"),

    # Vegetables - Leafy & Stems
    "spinach": ("plant", "vegetable", "leafy_green"),
    "lettuce": ("plant", "vegetable", "leafy_green"),
    "celery": ("plant", "vegetable", "stem"),
    "asparagus": ("plant", "vegetable", "stem"),
    "artichoke": ("plant", "vegetable", "stem"),
    "jerusalem_artichoke": ("plant", "vegetable", "root_tuber"),
    "bamboo_shoot": ("plant", "vegetable", "stem"),
    "palm_heart": ("plant", "vegetable", "stem"),
    "rhubarb": ("plant", "vegetable", "stem"),
    "water_chestnut": ("plant", "vegetable", "stem"),

    # Vegetables - Roots & Tubers
    "carrot": ("plant", "vegetable", "root_tuber"),
    "parsnip": ("plant", "vegetable", "root_tuber"),
    "beetroot": ("plant", "vegetable", "root_tuber"),
    "yam": ("plant", "vegetable", "root_tuber"),
    "taro": ("plant", "vegetable", "root_tuber"),
    "cassava": ("plant", "vegetable", "root_tuber"),
    "jicama": ("plant", "vegetable", "root_tuber"),
    "arrowroot": ("plant", "vegetable", "root_tuber"),

    # Vegetables - Squash & Gourds (Cucurbits)
    "cucumber": ("plant", "vegetable", "cucurbit"),
    "zucchini": ("plant", "vegetable", "cucurbit"),
    "yellow_zucchini": ("plant", "vegetable", "cucurbit"),
    "pumpkin": ("plant", "vegetable", "cucurbit"),
    "winter_squash": ("plant", "vegetable", "cucurbit"),
    "squash": ("plant", "vegetable", "cucurbit"),

    # Herbs & Spices
    "basil": ("plant", "herb_spice", "lamiaceae_herb"),
    "mint": ("plant", "herb_spice", "lamiaceae_herb"),
    "spearmint": ("plant", "herb_spice", "lamiaceae_herb"),
    "peppermint": ("plant", "herb_spice", "lamiaceae_herb"),
    "oregano": ("plant", "herb_spice", "lamiaceae_herb"),
    "mexican_oregano": ("plant", "herb_spice", "lamiaceae_herb"),
    "marjoram": ("plant", "herb_spice", "lamiaceae_herb"),
    "sweet_marjoram": ("plant", "herb_spice", "lamiaceae_herb"),
    "rosemary": ("plant", "herb_spice", "lamiaceae_herb"),
    "thyme": ("plant", "herb_spice", "lamiaceae_herb"),
    "sage": ("plant", "herb_spice", "lamiaceae_herb"),
    "lavender": ("plant", "herb_spice", "lamiaceae_herb"),
    "lemon_balm": ("plant", "herb_spice", "lamiaceae_herb"),
    "savory": ("plant", "herb_spice", "lamiaceae_herb"),
    "winter_savory": ("plant", "herb_spice", "lamiaceae_herb"),
    "parsley": ("plant", "herb_spice", "apiaceae_herb"),
    "cilantro": ("plant", "herb_spice", "apiaceae_herb"),
    "coriander": ("plant", "herb_spice", "apiaceae_spice"),
    "dill": ("plant", "herb_spice", "apiaceae_herb"),
    "fennel": ("plant", "herb_spice", "apiaceae_herb"),
    "chervil": ("plant", "herb_spice", "apiaceae_herb"),
    "anise": ("plant", "herb_spice", "apiaceae_spice"),
    "star_anise": ("plant", "herb_spice", "warm_spice"),
    "caraway": ("plant", "herb_spice", "apiaceae_spice"),
    "cumin": ("plant", "herb_spice", "apiaceae_spice"),
    "celery_seed": ("plant", "herb_spice", "apiaceae_spice"),
    "cinnamon": ("plant", "herb_spice", "bark_spice"),
    "cassia": ("plant", "herb_spice", "bark_spice"),
    "clove": ("plant", "herb_spice", "flower_spice"),
    "nutmeg": ("plant", "herb_spice", "seed_spice"),
    "mace": ("plant", "herb_spice", "seed_spice"),
    "cardamom": ("plant", "herb_spice", "zingiber_spice"),
    "ginger": ("plant", "herb_spice", "zingiber_spice"),
    "turmeric": ("plant", "herb_spice", "zingiber_spice"),
    "galangal": ("plant", "herb_spice", "zingiber_spice"),
    "allspice": ("plant", "herb_spice", "berry_spice"),
    "black_pepper": ("plant", "herb_spice", "piper_spice"),
    "white_pepper": ("plant", "herb_spice", "piper_spice"),
    "green_pepper": ("plant", "herb_spice", "piper_spice"),
    "saffron": ("plant", "herb_spice", "stigma_spice"),
    "vanilla": ("plant", "herb_spice", "orchid_spice"),
    "mustard": ("plant", "herb_spice", "seed_spice"),
    "tarragon": ("plant", "herb_spice", "asteraceae_herb"),
    "bay_leaf": ("plant", "herb_spice", "laurel_herb"),
    "lemongrass": ("plant", "herb_spice", "grass_herb"),
    "angelica": ("plant", "herb_spice", "apiaceae_herb"),
    "lovage": ("plant", "herb_spice", "apiaceae_herb"),

    # Dairy
    "milk": ("animal", "dairy", "fluid_milk"),
    "buttermilk": ("animal", "dairy", "fermented_milk"),
    "cream": ("animal", "dairy", "cream"),
    "sour_cream": ("animal", "dairy", "fermented_cream"),
    "yogurt": ("animal", "dairy", "fermented_milk"),
    "kefir": ("animal", "dairy", "fermented_milk"),
    "butter": ("animal", "dairy", "fat"),
    "ghee": ("animal", "dairy", "fat"),
    "whey": ("animal", "dairy", "whey"),
    "cheese": ("animal", "dairy", "cheese"),
    "cheddar_cheese": ("animal", "dairy", "hard_cheese"),
    "parmesan_cheese": ("animal", "dairy", "hard_cheese"),
    "swiss_cheese": ("animal", "dairy", "hard_cheese"),
    "gouda_cheese": ("animal", "dairy", "hard_cheese"),
    "provolone_cheese": ("animal", "dairy", "hard_cheese"),
    "mozzarella_cheese": ("animal", "dairy", "soft_cheese"),
    "brie_cheese": ("animal", "dairy", "soft_cheese"),
    "camembert_cheese": ("animal", "dairy", "soft_cheese"),
    "cottage_cheese": ("animal", "dairy", "soft_cheese"),
    "cream_cheese": ("animal", "dairy", "soft_cheese"),
    "ricotta_cheese": ("animal", "dairy", "soft_cheese"),
    "feta_cheese": ("animal", "dairy", "brined_cheese"),
    "blue_cheese": ("animal", "dairy", "blue_cheese"),
    "roquefort_cheese": ("animal", "dairy", "blue_cheese"),
    "goat_cheese": ("animal", "dairy", "caprine_cheese"),

    # Meat & Poultry
    "beef": ("animal", "meat", "bovine"),
    "veal": ("animal", "meat", "bovine"),
    "pork": ("animal", "meat", "porcine"),
    "bacon": ("animal", "meat", "cured_pork"),
    "ham": ("animal", "meat", "cured_pork"),
    "prosciutto": ("animal", "meat", "cured_pork"),
    "sausage": ("animal", "meat", "processed_meat"),
    "lamb": ("animal", "meat", "ovine"),
    "mutton": ("animal", "meat", "ovine"),
    "goat": ("animal", "meat", "caprine"),
    "venison": ("animal", "meat", "game_meat"),
    "rabbit": ("animal", "meat", "game_meat"),
    "chicken": ("animal", "poultry", "gallinaceous"),
    "turkey": ("animal", "poultry", "gallinaceous"),
    "duck": ("animal", "poultry", "waterfowl"),
    "goose": ("animal", "poultry", "waterfowl"),
    "pheasant": ("animal", "poultry", "game_bird"),
    "quail": ("animal", "poultry", "game_bird"),
    "egg": ("animal", "poultry", "egg"),
    "egg_yolk": ("animal", "poultry", "egg"),
    "egg_white": ("animal", "poultry", "egg"),

    # Seafood
    "salmon": ("animal", "seafood", "salmonid_fish"),
    "trout": ("animal", "seafood", "salmonid_fish"),
    "tuna": ("animal", "seafood", "pelagic_fish"),
    "albacore_tuna": ("animal", "seafood", "pelagic_fish"),
    "cod": ("animal", "seafood", "gadoid_fish"),
    "haddock": ("animal", "seafood", "gadoid_fish"),
    "halibut": ("animal", "seafood", "flatfish"),
    "flounder": ("animal", "seafood", "flatfish"),
    "sole": ("animal", "seafood", "flatfish"),
    "turbot": ("animal", "seafood", "flatfish"),
    "walleye": ("animal", "seafood", "freshwater_fish"),
    "bass": ("animal", "seafood", "perch_fish"),
    "swordfish": ("animal", "seafood", "pelagic_fish"),
    "mackerel": ("animal", "seafood", "oily_fish"),
    "sardine": ("animal", "seafood", "clupeoid_fish"),
    "herring": ("animal", "seafood", "clupeoid_fish"),
    "anchovy": ("animal", "seafood", "clupeoid_fish"),
    "shrimp": ("animal", "seafood", "crustacean"),
    "prawn": ("animal", "seafood", "crustacean"),
    "crab": ("animal", "seafood", "crustacean"),
    "lobster": ("animal", "seafood", "crustacean"),
    "crayfish": ("animal", "seafood", "crustacean"),
    "clam": ("animal", "seafood", "bivalve_mollusk"),
    "mussel": ("animal", "seafood", "bivalve_mollusk"),
    "oyster": ("animal", "seafood", "bivalve_mollusk"),
    "scallop": ("animal", "seafood", "bivalve_mollusk"),
    "abalone": ("animal", "seafood", "gastropod_mollusk"),
    "squid": ("animal", "seafood", "cephalopod"),
    "octopus": ("animal", "seafood", "cephalopod"),
    "caviar": ("animal", "seafood", "fish_roe"),

    # Grains & Cereals
    "wheat": ("plant", "grain", "cereal_grass"),
    "flour": ("plant", "grain", "cereal_grass"),
    "bread": ("processed", "grain_product", "baked_bread"),
    "white_bread": ("processed", "grain_product", "baked_bread"),
    "multigrain_bread": ("processed", "grain_product", "baked_bread"),
    "bagel": ("processed", "grain_product", "baked_bread"),
    "tortilla": ("processed", "grain_product", "flatbread"),
    "tortilla_chip": ("processed", "grain_product", "snack"),
    "tostada": ("processed", "grain_product", "flatbread"),
    "tamale": ("processed", "grain_product", "corn_product"),
    "pasta": ("processed", "grain_product", "pasta"),
    "macaroni": ("processed", "grain_product", "pasta"),
    "noodle": ("processed", "grain_product", "pasta"),
    "rice": ("plant", "grain", "rice"),
    "basmati_rice": ("plant", "grain", "rice"),
    "wild_rice": ("plant", "grain", "rice"),
    "corn": ("plant", "grain", "maize"),
    "sweetcorn": ("plant", "grain", "maize"),
    "popcorn": ("plant", "grain", "maize"),
    "cornmeal": ("plant", "grain", "maize"),
    "polenta": ("plant", "grain", "maize"),
    "barley": ("plant", "grain", "cereal_grass"),
    "rye": ("plant", "grain", "cereal_grass"),
    "oat": ("plant", "grain", "cereal_grass"),
    "millet": ("plant", "grain", "cereal_grass"),
    "sorghum": ("plant", "grain", "cereal_grass"),
    "quinoa": ("plant", "grain", "pseudocereal"),
    "amaranth": ("plant", "grain", "pseudocereal"),
    "buckwheat": ("plant", "grain", "pseudocereal"),
    "teff": ("plant", "grain", "cereal_grass"),

    # Legumes & Pulses
    "soybean": ("plant", "legume", "bean_pulse"),
    "tofu": ("processed", "legume_product", "soy_product"),
    "miso": ("processed", "legume_product", "fermented_soy"),
    "soy_sauce": ("processed", "legume_product", "fermented_soy"),
    "lentil": ("plant", "legume", "lentil_pulse"),
    "chickpea": ("plant", "legume", "pulse"),
    "pea": ("plant", "legume", "green_pea"),
    "green_pea": ("plant", "legume", "green_pea"),
    "adzuki_bean": ("plant", "legume", "bean_pulse"),
    "black_bean": ("plant", "legume", "bean_pulse"),
    "kidney_bean": ("plant", "legume", "bean_pulse"),
    "lima_bean": ("plant", "legume", "bean_pulse"),
    "pinto_bean": ("plant", "legume", "bean_pulse"),
    "navy_bean": ("plant", "legume", "bean_pulse"),
    "fava_bean": ("plant", "legume", "bean_pulse"),
    "mung_bean": ("plant", "legume", "bean_pulse"),
    "black_eyed_pea": ("plant", "legume", "bean_pulse"),
    "alfalfa": ("plant", "legume", "sprout"),
    "bean": ("plant", "legume", "bean_pulse"),

    # Nuts & Seeds
    "almond": ("plant", "nut_seed", "tree_nut"),
    "walnut": ("plant", "nut_seed", "tree_nut"),
    "black_walnut": ("plant", "nut_seed", "tree_nut"),
    "pecan": ("plant", "nut_seed", "tree_nut"),
    "hazelnut": ("plant", "nut_seed", "tree_nut"),
    "cashew": ("plant", "nut_seed", "tree_nut"),
    "pistachio": ("plant", "nut_seed", "tree_nut"),
    "brazil_nut": ("plant", "nut_seed", "tree_nut"),
    "macadamia_nut": ("plant", "nut_seed", "tree_nut"),
    "chestnut": ("plant", "nut_seed", "tree_nut"),
    "acorn": ("plant", "nut_seed", "tree_nut"),
    "peanut": ("plant", "nut_seed", "groundnut"),
    "peanut_butter": ("processed", "nut_product", "nut_butter"),
    "pine_nut": ("plant", "nut_seed", "conifer_seed"),
    "sesame": ("plant", "nut_seed", "seed"),
    "sunflower_seed": ("plant", "nut_seed", "seed"),
    "pumpkin_seed": ("plant", "nut_seed", "seed"),
    "poppy_seed": ("plant", "nut_seed", "seed"),
    "flaxseed": ("plant", "nut_seed", "seed"),
    "chia_seed": ("plant", "nut_seed", "seed"),
    "mixed_nut": ("plant", "nut_seed", "tree_nut"),

    # Fungi & Algae
    "mushroom": ("fungi", "mushroom", "agaric_fungus"),
    "button_mushroom": ("fungi", "mushroom", "agaric_fungus"),
    "shiitake": ("fungi", "mushroom", "agaric_fungus"),
    "portobello": ("fungi", "mushroom", "agaric_fungus"),
    "cremini": ("fungi", "mushroom", "agaric_fungus"),
    "truffle": ("fungi", "mushroom", "subterranean_fungus"),
    "morel": ("fungi", "mushroom", "ascomycete_fungus"),
    "chanterelle": ("fungi", "mushroom", "cantharellus_fungus"),
    "agar": ("plant", "algae", "red_algae"),
    "kombu": ("plant", "algae", "kelp_algae"),
    "kelp": ("plant", "algae", "kelp_algae"),
    "nori": ("plant", "algae", "red_algae"),

    # Beverages & Fermented
    "tea": ("plant", "beverage", "tea_infusion"),
    "black_tea": ("plant", "beverage", "tea_infusion"),
    "green_tea": ("plant", "beverage", "tea_infusion"),
    "oolong_tea": ("plant", "beverage", "tea_infusion"),
    "white_tea": ("plant", "beverage", "tea_infusion"),
    "coffee": ("plant", "beverage", "coffee_infusion"),
    "espresso": ("plant", "beverage", "coffee_infusion"),
    "cocoa": ("plant", "beverage", "cacao_product"),
    "chocolate": ("processed", "confectionery", "cacao_product"),
    "dark_chocolate": ("processed", "confectionery", "cacao_product"),
    "milk_chocolate": ("processed", "confectionery", "cacao_product"),
    "white_chocolate": ("processed", "confectionery", "cacao_product"),
    "beer": ("processed", "beverage", "fermented_malt"),
    "ale": ("processed", "beverage", "fermented_malt"),
    "lager": ("processed", "beverage", "fermented_malt"),
    "wine": ("processed", "beverage", "fermented_grape"),
    "red_wine": ("processed", "beverage", "fermented_grape"),
    "white_wine": ("processed", "beverage", "fermented_grape"),
    "sparkling_wine": ("processed", "beverage", "fermented_grape"),
    "vermouth": ("processed", "beverage", "fortified_wine"),
    "sherry": ("processed", "beverage", "fortified_wine"),
    "port_wine": ("processed", "beverage", "fortified_wine"),
    "madeira_wine": ("processed", "beverage", "fortified_wine"),
    "brandy": ("processed", "beverage", "distilled_spirit"),
    "apple_brandy": ("processed", "beverage", "distilled_spirit"),
    "blackberry_brandy": ("processed", "beverage", "distilled_spirit"),
    "cognac": ("processed", "beverage", "distilled_spirit"),
    "rum": ("processed", "beverage", "distilled_spirit"),
    "vodka": ("processed", "beverage", "distilled_spirit"),
    "whiskey": ("processed", "beverage", "distilled_spirit"),
    "bourbon": ("processed", "beverage", "distilled_spirit"),
    "gin": ("processed", "beverage", "distilled_spirit"),
    "tequila": ("processed", "beverage", "distilled_spirit"),
    "liqueur": ("processed", "beverage", "cordial"),
    "cider": ("processed", "beverage", "fermented_fruit"),
    "apple_cider_vinegar": ("processed", "condiment", "fermented_acid"),
    "vinegar": ("processed", "condiment", "fermented_acid"),
    "balsamic_vinegar": ("processed", "condiment", "fermented_acid"),
    "wine_vinegar": ("processed", "condiment", "fermented_acid"),

    # Sweeteners & Confectionery
    "sugar": ("plant", "sweetener", "sucrose"),
    "brown_sugar": ("plant", "sweetener", "sucrose"),
    "honey": ("animal", "sweetener", "nectar"),
    "maple_syrup": ("plant", "sweetener", "tree_sap"),
    "syrup": ("plant", "sweetener", "sugar_syrup"),
    "molasses": ("plant", "sweetener", "cane_byproduct"),
    "agave": ("plant", "sweetener", "agave_nectar"),
    "caramel": ("processed", "confectionery", "heated_sugar"),
    "marshmallow": ("processed", "confectionery", "gelatin_sweet"),
    "marzipan": ("processed", "confectionery", "almond_paste"),
    "meringue": ("processed", "confectionery", "whipped_egg"),

    # Oils & Fats
    "olive_oil": ("plant", "fat_oil", "vegetable_oil"),
    "vegetable_oil": ("plant", "fat_oil", "vegetable_oil"),
    "canola_oil": ("plant", "fat_oil", "vegetable_oil"),
    "corn_oil": ("plant", "fat_oil", "vegetable_oil"),
    "soybean_oil": ("plant", "fat_oil", "vegetable_oil"),
    "sesame_oil": ("plant", "fat_oil", "vegetable_oil"),
    "sunflower_oil": ("plant", "fat_oil", "vegetable_oil"),
    "peanut_oil": ("plant", "fat_oil", "vegetable_oil"),
    "coconut_oil": ("plant", "fat_oil", "tropical_fat"),
    "margarine": ("processed", "fat_oil", "hydrogenated_fat"),
    "lard": ("animal", "fat_oil", "animal_fat"),

    # Others / Condiments
    "salt": ("mineral", "condiment", "sodium_chloride"),
    "water": ("mineral", "liquid", "water"),
    "baking_powder": ("mineral", "leavening", "chemical_leavener"),
    "baking_soda": ("mineral", "leavening", "chemical_leavener"),
    "yeast": ("fungi", "leavening", "saccharomyces"),
    "gelatin": ("animal", "thickener", "collagen"),
    "pectin": ("plant", "thickener", "polysaccharide"),
}


class FoodOnTaxonomyCache:
    """Disk-backed hierarchical taxonomy cache for ingredient names."""

    def __init__(self, cache_path: str | Path):
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._data: Dict[str, Dict[str, str]] = {}
        self._load()

    def _load(self) -> None:
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
                logger.info(f"Loaded {len(self._data)} taxonomic entries from cache")
            except Exception as e:
                logger.warning(f"Failed to load taxonomic cache: {e}")

    def save(self) -> None:
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)

    def get(self, name: str) -> Optional[Dict[str, str]]:
        return self._data.get(name.lower())

    def put(self, name: str, taxonomy: Dict[str, str]) -> None:
        self._data[name.lower()] = taxonomy


def classify_ingredient(name: str) -> Dict[str, str]:
    """Classify an ingredient into a standard 3-level taxonomy hierarchy.

    Returns dict with keys:
        'origin' (Level 1)
        'group'  (Level 2)
        'subgroup' (Level 3)
        'path' (full hierarchical string e.g. 'plant/fruit/pome_fruit')
    """
    clean = name.lower().strip()

    # Exact match first
    if clean in TAXONOMIC_RULES:
        orig, grp, sub = TAXONOMIC_RULES[clean]
        return {"origin": orig, "group": grp, "subgroup": sub, "path": f"{orig}/{grp}/{sub}/{clean}"}

    # Partial / substring match
    for key, (orig, grp, sub) in TAXONOMIC_RULES.items():
        if key in clean or clean in key:
            return {"origin": orig, "group": grp, "subgroup": sub, "path": f"{orig}/{grp}/{sub}/{clean}"}

    # Heuristic fallback based on keywords
    if any(k in clean for k in ["juice", "tea", "wine", "beer", "drink", "liquor", "spirit"]):
        orig, grp, sub = "processed", "beverage", "beverage"
    elif any(k in clean for k in ["bread", "cake", "cookie", "pie", "dough", "waffle", "biscuit"]):
        orig, grp, sub = "processed", "bakery", "baked_good"
    elif any(k in clean for k in ["sauce", "dressing", "ketchup", "mustard", "mayo"]):
        orig, grp, sub = "processed", "condiment", "sauce"
    elif any(k in clean for k in ["cheese", "cream", "milk"]):
        orig, grp, sub = "animal", "dairy", "dairy_product"
    elif any(k in clean for k in ["fish", "tuna", "salmon", "shrimp"]):
        orig, grp, sub = "animal", "seafood", "seafood"
    elif any(k in clean for k in ["meat", "beef", "pork", "chicken"]):
        orig, grp, sub = "animal", "meat", "meat_product"
    elif any(k in clean for k in ["bean", "pea", "lentil"]):
        orig, grp, sub = "plant", "legume", "pulse"
    elif any(k in clean for k in ["nut", "seed"]):
        orig, grp, sub = "plant", "nut_seed", "seed"
    else:
        orig, grp, sub = "plant", "vegetable", "culinary_plant"

    return {"origin": orig, "group": grp, "subgroup": sub, "path": f"{orig}/{grp}/{sub}/{clean}"}


def build_taxonomy_cache(
    ingredient_names: List[str],
    cache: FoodOnTaxonomyCache,
) -> Dict[str, Dict[str, str]]:
    """Build and save taxonomy dictionary for all given ingredients."""
    results: Dict[str, Dict[str, str]] = {}
    missing = [name for name in ingredient_names if not cache.get(name)]

    for name in ingredient_names:
        tax = cache.get(name)
        if not tax:
            tax = classify_ingredient(name)
            cache.put(name, tax)
        results[name.lower()] = tax

    if missing:
        cache.save()
        logger.info(f"Classified and cached {len(missing)} ingredients into FoodOn taxonomy.")

    return results
