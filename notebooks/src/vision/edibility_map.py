# edibility_map.py — Binary edibility labels for the 45 shared species across datasets
#
# Used by dataset.py to label images
#
# Sources: MycoBank, GBIF toxicity records, Roger Phillips Mushrooms (2006),
#          First Nature (firstnature.com), mycological consensus literature.
#
# Binary label convention
#   0 = edible
#   1 = poisonous (includes deadly, toxic, and conditionally edible)

#
# Each entry: "Species name": ("edible" or "poisonous", binary_label(0 for edible, 1 for poisonous))

EDIBILITY_MAP = {
    # AGARICUS 
    "Agaricus augustus": ("edible", 0),
    "Agaricus campestris": ("edible", 0),
    "Agaricus xanthodermus": ("poisonous", 1),

    # AMANITA 
    # Amanita phalloides (Death Cap) and A. virosa (Destroying Angel) cause the majority of fatal mushroom poisonings worldwide via amatoxins.
    "Amanita citrina": ("poisonous", 1,),
    "Amanita muscaria": ("poisonous", 1),
    "Amanita pantherina": ("poisonous", 1),
    "Amanita phalloides": ("poisonous", 1),
    "Amanita rubescens": ("poisonous", 1),
    "Amanita virosa": ("poisonous", 1),

    # ARMILLARIA 
    # Edible only when well cooked
    "Armillaria mellea":("poisonous", 1),

    # BOLETUS 
    "Boletus edulis": ("edible", 0),

    # CANTHARELLUS 
    "Cantharellus cibarius":("edible", 0),

    # CLITOCYBE 
    "Clitocybe nebularis": ("poisonous", 1),
    "Clitocybe odora": ("edible", 0),

    # COPRINUS
    "Coprinus comatus": ("edible", 0),

    # CORTINARIUS
    # Orellanine toxin causes delayed kidney failure
    "Cortinarius rubellus": ("poisonous", 1),

    # CRATERELLUS 
    "Craterellus cornucopioides": ("edible", 0),

    # FLAMMULINA 
    "Flammulina velutipes": ("edible", 0),

    # GALERINA 
    # Classic deadly look-alike — visually similar to edible honey fungus
    "Galerina marginata": ("poisonous", 1),

    # GRIFOLA 
    "Grifola frondosa": ("edible", 0),

    # GYROMITRA 
    # Gyromitrin causes liver damage and can be fatal.
    "Gyromitra esculenta": ("poisonous", 1),

    # HEBELOMA 
    "Hebeloma crustuliniforme": ("poisonous", 1),

    # HYDNUM 
    "Hydnum repandum": ("edible", 0),

    # HYGROCYBE 
    "Hygrocybe conica": ("poisonous", 1),

    # HYPHOLOMA 
    "Hypholoma fasciculare": ("poisonous", 1),

    # INOCYBE 
    "Inocybe geophylla": ("poisonous", 1),

    # LACCARIA 
    "Laccaria amethystina": ("edible", 0),
    "Laccaria laccata": ("edible", 0),

    # LACTARIUS 
    "Lactarius deliciosus": ("edible", 0),
    "Lactarius torminosus": ("poisonous", 1),

    # LEPIOTA 
    "Lepiota cristata": ("poisonous", 1),

    # LEPISTA 
    "Lepista nuda": ("edible",0),

    # MACROLEPIOTA 
    "Macrolepiota procera": ("edible", 0),

    # MARASMIUS 
    "Marasmius oreades": ("edible", 0),

    # MORCHELLA 
    "Morchella esculenta": ("edible", 0),

    # MYCENA 
    "Mycena galericulata":("poisonous", 1),

    # PAXILLUS
    # Causes immunohaemolytic syndrome
    "Paxillus involutus": ("poisonous", 1),

    # PLEUROTUS 
    "Pleurotus ostreatus": ("edible", 0),

    # RUSSULA
    "Russula cyanoxantha": ("edible", 0),
    "Russula emetica": ("poisonous", 1),
    "Russula ochroleuca": ("poisonous", 1),

    # SUILLUS
    "Suillus luteus": ("edible", 0),

    # TRICHOLOMA 
    "Tricholoma equestre": ("poisonous", 1),
    "Tricholoma portentosum": ("edible", 0),
    "Tricholoma terreum": ("edible", 0),
}

# Returns binary label: 0 = edible, 1 = poisonous, -1 = unknown species.
# -1 causes dataset.py to skip the sample entirely.
def get_label(species: str) -> int:
    entry = EDIBILITY_MAP.get(species)
    return entry[1] if entry else -1

# Prints a count of edible vs poisonous species and class balance.
def summary() -> None:
    from collections import Counter
    labels = Counter(lbl for _, lbl in EDIBILITY_MAP.values())
    cats = Counter(cat for cat, _ in EDIBILITY_MAP.values())
    print(f"Total species: {len(EDIBILITY_MAP)}")
    print(f"edible: {cats['edible']}")
    print(f"poisonous: {cats['poisonous']}")
    print(f"Binary balance: {labels[0]/len(EDIBILITY_MAP)*100:.1f}% edible / "
          f"{labels[1]/len(EDIBILITY_MAP)*100:.1f}% poisonous")


if __name__ == "__main__":
    summary()