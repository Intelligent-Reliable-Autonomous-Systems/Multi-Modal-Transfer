import numpy as np

# Available cultivars for simulation
CROP_NAMES = {
    "grape_phenology_": np.array(
        [
            "Aligote",
            "Alvarinho",
            "Auxerrois",
            "Barbera",
            "Cabernet_Franc",
            "Cabernet_Sauvignon",
            "Chardonnay",
            "Chenin_Blanc",
            "Concord",
            "Durif",
            "Gewurztraminer",
            "Green_Veltliner",
            "Grenache",
            "Lemberger",
            "Malbec",
            "Melon",
            "Merlot",
            "Mourvedre",
            "Muscat_Blanc",
            "Nebbiolo",
            "Petit_Verdot",
            "Pinot_Blanc",
            "Pinot_Gris",
            "Pinot_Noir",
            "Riesling",
            "Sangiovese",
            "Sauvignon_Blanc",
            "Semillon",
            "Tempranillo",
            "Viognier",
            "Zinfandel",
        ],
        dtype=str,
    ),
    "grape_coldhardiness_ferg": np.array(
        [
            "Barbera",
            "Cabernet_Franc",
            "Cabernet_Sauvignon",
            "Chardonnay",
            "Chenin_Blanc",
            "Concord",
            "Gewurztraminer",
            "Grenache",
            "Lemberger",
            "Malbec",
            "Merlot",
            "Mourvedre",
            "Nebbiolo",
            "Pinot_Gris",
            "Riesling",
            "Sangiovese",
            "Sauvignon_Blanc",
            "Semillon",
            "Syrah",
            "Viognier",
            "Zinfandel",
        ],
        dtype=str,
    ),
    "grape_coldhardiness_reg": np.array(
        [
            "Cabernet_Franc",  # 0
            "Cabernet_Sauvignon",  # 1
            "Chardonnay",  # 2
            "Gewurztraminer",  # 3
            "Merlot",  # 4
            "Pinot_Blanc",  # 5
            "Pinot_Gris",  # 6
            "Pinot_Noir",  # 7
            "Riesling",  # 8
            "Sauvignon_Blanc",  # 9
            "Syrah",  # 10
            "Zinfandel",  # 11
            # "Barbera",  # 12
            # "Chenin_Blanc",  # 13
            # "Concord",  # 14
            # "Grenache",  # 15
            # "Lemberger",  # 16
            # "Malbec",  # 17
            # "Mourvedre",  # 18
            # "Nebbiolo",  # 19
            # "Sangiovese",  # 20
            # "Semillon",  # 21
            # "Viognier",  # 22
        ],
        dtype=str,
    ),
    "grape_coldhardiness_all": np.array(
        [
            "Aligote",
            "Alvarinho",
            "Auxerrois",
            "Barbera",
            "Cabernet_Franc",
            "Cabernet_Sauvignon",
            "Chardonnay",
            "Chenin_Blanc",
            "Concord",
            "Gewurztraminer",
            "Grenache",
            "Lemberger",
            "Malbec",
            "Melon",
            "Merlot",
            "Mourvedre",
            "Muscat_Blanc",
            "Nebbiolo",
            "Pinot_Blanc",
            "Pinot_Gris",
            "Pinot_Noir",
            "Riesling",
            "Sangiovese",
            "Sauvignon_Blanc",
            "Semillon",
            "Syrah",
            "Viognier",
            "Zinfandel",
        ],
        dtype=str,
    ),
    "wofost_": np.array(
        [
            "Winter_Wheat_101",
            "Winter_Wheat_102",
            "Winter_Wheat_103",
            "Winter_Wheat_104",
            "Winter_Wheat_105",
            "Winter_Wheat_106",
            "Winter_Wheat_107",
            "Bermude",
            "Apache",
        ],
        dtype=str,
    ),
    "cherry_coldhardiness_ferg": np.array(
        [
            "Attika",
            "Benton",
            "Bing",
            "Black_Gold",
            "Black_Pearl",
            "Black_Tartarian",
            "Chelan",
            "Early_Robin",
            "Early_Star",
            # "Emperor_Francis",
            "Hedelfingen",
            # "Index",
            "Kristen",
            "Lambert",
            "Lapins",
            "Napoleon",
            "Rainier",
            "Regina",
            "Sandra_Rose",
            "Santina",
            "Schmidt",
            "Schneider",
            "Selah",
            "Skeena",
            "Stella",
            "Sunburst",
            "Sweetheart",
            "Tieton",
            "Ulster",
            "Van",
            "Vega",
        ],
        dtype=str,
    ),
}

REGIONS = np.array(["BCOV", "ONNP", "WA", "NY", "NS", "MI", "QC", "WI", "TX", "PA"])
STATIONS = np.array(
    [
        "Kelowna",
        "Kelowna_UBC",
        "Osoyoos_CS",
        "Summerland",
        "Grimbsy_Mtn",
        "Vineland_Station",
        "Roza2",
        "SWMREC",
        "Beaver",
        "Bejo",
        "Geneva",
        "Hudson",
        "Portland",
        "Montreal",
        "Madison",
        "Brownfield",
        "PennState",
    ]
)
SITES = np.array(
    [
        "WestKelowna",  # 0
        "OKFallsEast",  # 1
        "OKFallsWest",  # 2
        "OliverEast",  # 3
        "OliverSouth",  # 4
        "OliverWest",  # 5
        "BlackSage",  # 6
        "OsoyoosNorth",  # 7
        "OsoyoosNortheast",  # 8
        "OsoyoosSoutheast",  # 9
        "OsoyoosWest",  # 10
        "NaramataBench",  # 11
        "BeamsvilleBench",  # 12
        "LincolnLakeshore",  # 13
        "TwentyMileBench",  # 14
        "VinemountRidge",  # 15
        "CreekShores",  # 16
        "FourMileCreek",  # 17
        "NiagaraLakeshores",  # 18
        "NiagaraRiver",  # 19
        "ShortHillsBench",  # 20
        "StDavidsBench",  # 21
        "Prosser",  # 22
        "Mich1",
        "Beaver",
        "Bejo",
        "Geneva",
        "Hudson",
        "Portland",
        "Montreal",
        "Madison",
        "Brownfield",
        "PennState",
    ]
)

REGION_NAMES = {
    "BCOV": {
        "Kelowna": ["WestKelowna"],
        "Kelowna_UBC": ["OKFallsEast", "OKFallsWest", "OliverEast", "OliverSouth", "OliverWest"],
        "Osoyoos_CS": ["BlackSage", "OsoyoosNorth", "OsoyoosNortheast", "OsoyoosSoutheast", "OsoyoosWest"],
        "Summerland": ["NaramataBench"],
    },
    "ONNP": {
        "Grimbsy_Mtn": ["BeamsvilleBench", "LincolnLakeshore", "TwentyMileBench", "VinemountRidge"],
        "Vineland_Station": [
            "CreekShores",
            "FourMileCreek",
            "NiagaraLakeshores",
            "NiagaraRiver",
            "ShortHillsBench",
            "StDavidsBench",
        ],
    },
    "WA": {"Roza2": ["Prosser"]},
}

CULT_STR = np.array(
    [
        "query: Cabernet Franc",
        "query: Cabernet Sauvignon",
        "query: Chardonnay",
        "query: Gewurztraminer",
        "query: Merlot",
        "query: Pinot Blanc",
        "query: Pinot Gris",
        "query: Pinot Noir",
        "query: Riesling",
        "query: Sauvignon Blanc",
        "query: Syrah",
        "query: Zinfandel",
        "query: Barbera",
        "query: Chenin Blanc",
        "query: Concord",
        "query: Grenache",
        "query: Lemberger",
        "query: Malbec",
        "query: Mourvedre",
        "query: Nebbiolo",
        "query: Sangiovese",
        "query: Semillon",
        "query: Viognier",
    ]
)

CULT_STR_VERB = np.array(
    [
        "query: Cabernet Franc is an early- to mid-budding, mid-ripening red grape cultivar with medium vigor and loose to moderately compact clusters, producing wines with moderate acidity and a lighter, translucent red color. It has good cold-hardiness (tolerating around -15C/5F) and prefers well-drained soils such as gravelly loam or limestone.",
        "query: Cabernet Sauvignon is a late-budding, late-ripening red grape cultivar with small, compact clusters, producing deeply colored wines with firm acidity and strong tannins. It has moderate cold-hardiness (tolerating around -15C/5F) and prefers well-drained gravelly or sandy-loam soils.",
        "query: Chardonnay is an early- to mid-ripening white grape cultivar with moderate vigor and medium-sized, loosely packed clusters, producing wines with bright acidity and a pale to golden color. It has moderate cold-hardiness (tolerating around -15C/5F) and prefers well-drained soils such as limestone, chalk, or clay.",
        "query: Gewurztraminer is an early- to mid-ripening white grape cultivar with low to moderate vigor and tightly packed clusters, producing intensely aromatic wines with low to moderate acidity and a golden to deep yellow color. It has low to moderate cold-hardiness (tolerating around -12C/10F) and prefers well-drained, calcareous or loamy soils.",
        "query: Merlot is a mid-budding, mid- to late-ripening red grape cultivar with medium vigor and medium-sized, loosely packed clusters, producing wines with soft tannins, moderate acidity, and a deep ruby color. It has moderate cold-hardiness (tolerating around -15C/5F) and prefers well-drained clay or gravelly soils.",
        "query: Pinot Blanc is an early- to mid-ripening white grape cultivar with moderate vigor and medium-sized, loosely packed clusters, producing wines with moderate acidity and a pale yellow to straw color. It has moderate cold-hardiness (tolerating around -15C/5F) and prefers well-drained, calcareous or loamy soils.",
        "query: Pinot Gris is an early- to mid-ripening white grape cultivar with moderate vigor and medium-sized, loosely packed clusters, producing wines with medium acidity and a color ranging from pale yellow to coppery gray. It has moderate cold-hardiness (tolerating around -15C/5F) and prefers well-drained, fertile soils such as loam or limestone.",
        "query: Pinot Noir is an early-budding, mid- to late-ripening red grape cultivar with low to moderate vigor and small, loosely packed clusters, producing wines with bright acidity and a light to medium ruby color. It has moderate cold-hardiness (tolerating around -15C/5F) and prefers well-drained, calcareous or loamy soils.",
        "query: Riesling is an early- to mid-budding, late-ripening white grape cultivar with moderate vigor and small to medium, compact clusters, producing wines with high acidity and a pale yellow to golden color. It has good cold-hardiness (tolerating around -20C/-4F) and prefers well-drained soils such as slate, limestone, or loam.",
        "query: Sauvignon Blanc is an early- to mid-ripening white grape cultivar with moderate vigor and medium-sized, loose clusters, producing wines with high acidity and a pale greenish-yellow color. It has moderate cold-hardiness (tolerating around -15C/5F) and prefers well-drained soils such as gravel, limestone, or sandy loam.",
        "query: Syrah is a mid- to late-budding, mid- to late-ripening red grape cultivar with high vigor and medium to large, compact clusters, producing wines with moderate acidity and a deep, dark purple color. It has moderate cold-hardiness (tolerating around -15C/5F) and prefers well-drained, stony or gravelly soils.",
        "query: Zinfandel is a mid- to late-budding, mid- to late-ripening red grape cultivar with high vigor and medium to large, loosely packed clusters, producing wines with moderate acidity and a deep ruby to purple color. It has low to moderate cold-hardiness (tolerating around -12C/10F) and prefers well-drained, sandy or loamy soils.",
        "query: Barbera is a mid-budding, mid- to late-ripening red grape cultivar with high vigor and medium to large, loosely packed clusters, producing wines with very high acidity and a deep ruby to purple color. It has moderate cold-hardiness (tolerating around -15C/5F) and prefers well-drained soils such as clay, limestone, or sandy loam.",
        "query: Chenin Blanc is an early- to mid-budding, mid-ripening white grape cultivar with moderate to high vigor and medium-sized, compact clusters, producing wines with high acidity and a pale yellow to golden color. It has moderate cold-hardiness (tolerating around -15C/5F) and prefers well-drained soils such as tuffeau, schist, or sandy loam.",
        "query: Concord is a mid-budding, early- to mid-ripening red grape cultivar with high vigor and medium to large, loosely packed clusters, producing wines with moderate acidity and a deep blue-purple color. It has excellent cold-hardiness (tolerating around -25C/-13F) and prefers well-drained soils such as sandy loam, loam, or glacial till.",
        "query: Grenache is a late-budding, late-ripening red grape cultivar with high vigor and medium to large, loosely packed clusters, producing wines with low acidity and a pale to medium ruby color. It has low to moderate cold-hardiness (tolerating around -12C/10F) and prefers well-drained soils such as sandy, stony, or gravelly loam.",
        "query: Lemberger is an early- to mid-budding, mid-ripening red grape cultivar with moderate vigor and medium-sized, moderately compact clusters, producing wines with moderate to high acidity and a deep blue-red color. It has good cold-hardiness (tolerating around -20C/-4F) and prefers well-drained soils such as loam, gravel, or limestone.",
        "query: Malbec is a mid-budding, mid- to late-ripening red grape cultivar with moderate to high vigor and medium-sized, compact clusters, producing wines with moderate acidity and a deep violet to inky purple color. It has moderate cold-hardiness (tolerating around -15C/5F) and prefers well-drained soils such as clay, limestone, or alluvial gravel.",
        "query: Mourvedre is a late-budding, late-ripening red grape cultivar with moderate vigor and small to medium, compact clusters, producing wines with moderate acidity and firm tannins and a deep ruby to purple color. It has low cold-hardiness (tolerating around -10C/14F) and prefers well-drained soils such as sandy, gravelly, or stony loam.",
        "query: Nebbiolo is an early-budding, late-ripening red grape cultivar with moderate vigor and small to medium, compact clusters, producing wines with very high acidity and strong tannins and a pale garnet to brick-red color. It has moderate cold-hardiness (tolerating around -15C/5F) and prefers well-drained soils such as calcareous marl, limestone, or sandy loam.",
        "query: Sangiovese is a mid-budding, mid- to late-ripening red grape cultivar with moderate to high vigor and medium-sized, compact clusters, producing wines with high acidity and firm tannins and a medium ruby to garnet color. It has moderate cold-hardiness (tolerating around -15C/5F) and prefers well-drained soils such as galestro, limestone, or sandy clay.",
        "query: Semillon is an early- to mid-ripening white grape cultivar with moderate vigor and medium to large, loosely packed clusters, producing wines with moderate acidity and a pale golden to deep yellow color. It has moderate cold-hardiness (tolerating around -15C/5F) and prefers well-drained soils such as gravel, sandy loam, or clay.",
        "query: Viognier is a mid-budding, mid-ripening white grape cultivar with low to moderate vigor and small to medium, loosely packed clusters, producing wines with low to moderate acidity and a deep golden to amber color. It has low to moderate cold-hardiness (tolerating around -12C/10F) and prefers well-drained soils such as granite, sandy loam, or gravelly schist.",
    ]
)

CULT_STR_MINI = np.array(
    [
        "query: Cabernet Franc is a mid-ripening red cultivar with moderate acidity and good cold-hardiness, preferring well-drained gravelly or limestone soils.",
        "query: Cabernet Sauvignon is a late-ripening red cultivar with firm acidity and strong tannins, preferring well-drained gravelly or sandy-loam soils.",
        "query: Chardonnay is a mid-ripening white cultivar with bright acidity, preferring well-drained limestone, chalk, or clay soils.",
        "query: Gewurztraminer is a mid-ripening aromatic white cultivar with low acidity and low cold-hardiness, preferring well-drained calcareous or loamy soils.",
        "query: Merlot is a mid-ripening red cultivar with soft tannins and moderate acidity, preferring well-drained clay or gravelly soils.",
        "query: Pinot Blanc is a mid-ripening white cultivar with moderate acidity, preferring well-drained calcareous or loamy soils.",
        "query: Pinot Gris is a mid-ripening white cultivar with medium acidity, preferring well-drained loam or limestone soils.",
        "query: Pinot Noir is a mid-ripening red cultivar with bright acidity and light color, preferring well-drained calcareous or loamy soils.",
        "query: Riesling is a late-ripening white cultivar with high acidity and good cold-hardiness, preferring well-drained slate, limestone, or loam soils.",
        "query: Sauvignon Blanc is a mid-ripening white cultivar with high acidity, preferring well-drained gravel, limestone, or sandy loam soils.",
        "query: Syrah is a late-ripening red cultivar with high vigor and moderate acidity, preferring well-drained stony or gravelly soils.",
        "query: Zinfandel is a late-ripening red cultivar with moderate acidity and low cold-hardiness, preferring well-drained sandy or loamy soils.",
        "query: Barbera is a late-ripening red cultivar with very high acidity, preferring well-drained clay, limestone, or sandy loam soils.",
        "query: Chenin Blanc is a mid-ripening white cultivar with high acidity, preferring well-drained tuffeau, schist, or sandy loam soils.",
        "query: Concord is an early-ripening red cultivar with excellent cold-hardiness, preferring well-drained sandy loam or loam soils.",
        "query: Grenache is a late-ripening red cultivar with low acidity and low cold-hardiness, preferring well-drained sandy or gravelly soils.",
        "query: Lemberger is a mid-ripening red cultivar with moderate-high acidity and good cold-hardiness, preferring well-drained loam, gravel, or limestone soils.",
        "query: Malbec is a late-ripening red cultivar with moderate acidity and deep color, preferring well-drained clay, limestone, or alluvial gravel soils.",
        "query: Mourvedre is a late-ripening red cultivar with firm tannins and low cold-hardiness, preferring well-drained sandy or gravelly soils.",
        "query: Nebbiolo is a late-ripening red cultivar with very high acidity, strong tannins, and pale color, preferring well-drained calcareous or limestone soils.",
        "query: Sangiovese is a late-ripening red cultivar with high acidity and firm tannins, preferring well-drained galestro, limestone, or sandy clay soils.",
        "query: Semillon is a mid-ripening white cultivar with moderate acidity, preferring well-drained gravel, sandy loam, or clay soils.",
        "query: Viognier is a mid-ripening white cultivar with low acidity and low cold-hardiness, preferring well-drained granite or sandy loam soils.",
    ]
)

CULT_STR_BARE = np.array(
    [
        "query: Cabernet Franc, red, mid-ripening, moderate acidity, good cold-hardiness, gravelly loam or limestone soils.",
        "query: Cabernet Sauvignon, red, late-ripening, firm acidity, strong tannins, gravelly or sandy-loam soils.",
        "query: Chardonnay, white, mid-ripening, bright acidity, limestone or clay soils.",
        "query: Gewurztraminer, white, mid-ripening, low acidity, aromatic, low cold-hardiness, calcareous or loamy soils.",
        "query: Merlot, red, mid-ripening, soft tannins, moderate acidity, clay or gravelly soils.",
        "query: Pinot Blanc, white, mid-ripening, moderate acidity, calcareous or loamy soils.",
        "query: Pinot Gris, white, mid-ripening, medium acidity, loam or limestone soils.",
        "query: Pinot Noir, red, mid-ripening, bright acidity, light color, calcareous or loamy soils.",
        "query: Riesling, white, late-ripening, high acidity, good cold-hardiness, slate or limestone soils.",
        "query: Sauvignon Blanc, white, mid-ripening, high acidity, gravel or limestone soils.",
        "query: Syrah, red, late-ripening, high vigor, moderate acidity, stony or gravelly soils.",
        "query: Zinfandel, red, late-ripening, moderate acidity, low cold-hardiness, sandy or loamy soils.",
        "query: Barbera, red, late-ripening, very high acidity, clay or limestone soils.",
        "query: Chenin Blanc, white, mid-ripening, high acidity, tuffeau or schist soils.",
        "query: Concord, red, early-ripening, excellent cold-hardiness, sandy loam or loam soils.",
        "query: Grenache, red, late-ripening, low acidity, low cold-hardiness, sandy or gravelly soils.",
        "query: Lemberger, red, mid-ripening, moderate-high acidity, good cold-hardiness, loam or gravel soils.",
        "query: Malbec, red, late-ripening, moderate acidity, deep color, clay or limestone soils.",
        "query: Mourvedre, red, late-ripening, firm tannins, low cold-hardiness, sandy or gravelly soils.",
        "query: Nebbiolo, red, late-ripening, very high acidity, strong tannins, calcareous or limestone soils.",
        "query: Sangiovese, red, late-ripening, high acidity, firm tannins, galestro or limestone soils.",
        "query: Semillon, white, mid-ripening, moderate acidity, gravel or sandy loam soils.",
        "query: Viognier, white, mid-ripening, low acidity, low cold-hardiness, granite or sandy loam soils.",
    ]
)

CULT_STR_TEST = np.array(
    [
        "query: a wooden spoon develops a rich patina over years of regular cooking use.",
        "query: a grandfather clock requires careful calibration to keep accurate time.",
        "query: a rubber band loses its elasticity when exposed to extreme heat.",
        "query: a ceramic mug retains heat far longer than a paper cup.",
        "query: a bicycle chain needs lubrication every few hundred miles of riding.",
        "query: a glass marble can refract light into a tiny rainbow on a sunny day.",
        "query: a wool blanket becomes softer with each gentle wash cycle.",
        "query: a cast iron skillet can last for several generations if properly seasoned.",
        "query: a compass needle always aligns itself with the Earth's magnetic field.",
        "query: a leather wallet gradually molds to the shape of its owner's pocket.",
        "query: a piano has exactly 88 keys spanning over seven octaves.",
        "query: a candle burns faster in a room with good air circulation.",
        "query: a tin can will begin to rust within weeks if left out in the rain.",
        "query: a tennis ball loses its bounce after prolonged exposure to cold temperatures.",
        "query: a cork expands slightly when inserted into a wine bottle.",
        "query: a matchstick contains a small amount of phosphorus at its tip.",
        "query: a paper clip can hold roughly ten sheets of standard printer paper.",
        "query: a linen shirt becomes more breathable after several washes.",
        "query: a pocket watch must be wound daily to maintain accurate timekeeping.",
        "query: a snow globe contains glycerin to slow the falling of the artificial snow.",
        "query: a wooden pencil can draw a line approximately 35 miles long before running out.",
        "query: a thermometer measures temperature by tracking the expansion of liquid inside.",
        "query: a doorknob made of brass naturally kills most bacteria within a few hours.",
    ]
)

SITE_STR_VERB = np.array(
    [
        "query: The West Kelowna site in the Okanagan Valley, British Columbia (approximately 49.86 N, -119.58 W) is a semi-arid, rain-shadow viticultural area with hot, sunny summers and cold winters (low annual precipitation ~350 mm) that support a long, intense growing season and moderate frost risk, and well-drained, mineral-rich soils including volcanic rock, glacial till, sandy loam and clay derived from ancient volcanic and glacial deposits that favor diverse grape varieties and concentrated fruit.",  # 0
        "query: The Okanagan Falls East site in the Okanagan Valley, British Columbia (approximately 49.35 N, -119.57 W) features a semi-arid, lake-moderated climate with warm summer days and cool nights that preserve acidity through a long, steady ripening season and relatively low frost risk, and is underpinned by diverse, well-drained soils including glaciofluvial gravel, sand, sandy loam and mineral-rich pockets derived from ancient glacial deposits that favor balanced vine growth and fruit quality.",  # 1
        "query: The Okanagan Falls West site in the Okanagan Valley, British Columbia (approximately 49.29 N, -119.53 W) sits on the western side of the Skaha Lake area just outside the Okanagan Falls sub-GI, where the climate is warm and semi-arid with hot, sunny days tempered by cool nights and lake-moderated temperatures that extend the growing season, and the soils are diverse, well-drained glaciofluvial deposits of gravel, sand and mineral-rich loams that promote balanced vine growth.",  # 2
        "query: The Oliver East site in the Okanagan Valley, British Columbia (approximately 49.18 N, -119.55 W) experiences a warm, semi-arid continental climate with long, hot, sunny summers and cool nights under a pronounced rain-shadow that supports a long growing season with strong diurnal shifts, and is underlain by well-drained, sandy, gravelly and glaciofluvial soils typical of the southern Okanagan that encourage deep root systems and concentrated fruit for a wide range of vinifera varieties.",  # 3
        "query: The Oliver South viticultural area in the southern Okanagan Valley, British Columbia (approximately 49.18 N, -119.55 W) lies in one of Canada's warmest, semi-arid, high-sun desert-influenced climates with hot, dry summers and cool nights that help maintain acidity in grapes, and is characterized by well-drained, glaciofluvial and sandy-gravel soils with low moisture retention that promote deep rooting and concentration in a broad range of varieties from reds to whites.",  # 4
        "query: The Oliver West site in the Okanagan Valley, British Columbia (approximately 49.18 N, -119.55 W) is part of a warm, semi-arid continental climate with hot, dry summers and cool nights under a pronounced rain-shadow that helps grapes ripen with concentration and preserve natural acidity, and is underlain by well-drained, glaciofluvial gravels, sands, and sandy loams typical of the South Okanagan that promote deep rooting and balanced vine growth for both red and white varieties.",  # 5
        "query: The Black Sage Bench site in the Okanagan Valley, British Columbia (approximately 49.06 N, 119.33 W) features a very warm, semi-arid “pocket desert” climate with hot, sunny summers and cool nights that enhance ripening and acidity retention, and is underlain by deep, well-drained sandy glaciofluvial soils that force vines to root deeply and produce concentrated fruit with minimal foliage.",  # 6
        "query: The Osoyoos North site in the Okanagan Valley, British Columbia (approximately 49.03 N, -119.46 W) lies just north of the town of Osoyoos in one of Canada's warmest, semi-arid, desert-influenced climates with hot, dry summers and cool nights that promote concentrated ripening and acidity retention, and is underlain by deep, well-drained sandy and glaciofluvial soils typical of the southern valley that encourage deep rooting and balanced vine development.",  # 7
        "query: The Osoyoos North East site in the Okanagan Valley, British Columbia (approximately 49.03 N, -119.46 W) lies in the semi-arid, rain-shadow-influenced southern Okanagan with hot, dry summers and cool nights that promote ripeness and acidity retention in grapes, and its soils are generally well-drained glaciofluvial sands, gravels, and mineral-rich loams typical of the broader Osoyoos/Oliver area that encourage deep root development and concentrated fruit character.",  # 8
        "query: The Osoyoos South East Bench site in the Okanagan Valley, British Columbia (approximately 49.03 N, -119.46 W, near the southeast flank of Osoyoos) experiences a hot, semi-arid, rain-shadow-influenced climate with long, dry, sunny summers and cool nights that promote ripening while preserving acidity, and is underlain by well-drained sandy, glaciofluvial soils with interspersed gravel and mineral-rich components typical of the southern valley that encourage deep vine rooting and concentrated fruit.",  # 9
        "query: The Osoyoos West site in the Okanagan Valley, British Columbia (approximately 49.03 N, -119.46 W) lies in a hot, semi-arid continental climate with long, dry, sunny summers and cooler nights that favor ripening and acidity retention and is underlain by well-drained, sandy glaciofluvial soils with traces of gravel and loam typical of the west side of Osoyoos Lake that encourage deep rooting and concentrated fruit.",  # 10
        "query: The Naramata Bench site in the Okanagan Valley, British Columbia (approximately 49.35 N, 119 35 W) has a moderate continental climate with warm, sunny summers and cool nights moderated by nearby Okanagan Lake that extend the growing season and preserve acidity, and is underlain by well-drained, mainly sandy to silt-loam glacial lake and mixed sediments with pockets of gravel and mineral-rich soils that encourage deep rooting and balanced vine growth.",  # 11
        "query: The Beamsville Bench site in the Ontario Niagara Peninsula (approximately 43.17 N, -79.50 W) has a cool-climate, lake-moderated growing season with warm, sunny days and cooling breezes off Lake Ontario that extend ripening and moderate frost risk, and is underlain by deep, well-drained, heterogeneous soils of gravel, sand, silt, clay and fossil-rich dolomitic limestone derived from the Niagara Escarpment that impart notable minerality and balanced moisture for vine growth.",  # 12
        "query: The Lincoln Lakeshore site in the Ontario Niagara Peninsula (approximately 43.11 N, -79.34 W) lies along the southern shore of Lake Ontario with a cool, lake-moderated climate that yields a long, temperate growing season with warm days and cool nights, and is underlain by well-drained, variable soils including light sandy soils and pockets of red clay loam over ancient Lake Iroquois plains that provide balanced moisture and heat for vine growth.",  # 13
        "query: The Twenty Mile Bench site in the Ontario Niagara Peninsula (approximately 43.15 N, -79.39 W) has a cool-climate, moderating influence from Lake Ontario with warm, sunny growing seasons and gentle temperature swings aided by breezes and escarpment shelter, and is underlain by deep, well-drained glacial clay and till soils rich in limestone and shale that retain moisture yet offer good drainage for balanced vine growth and fruit development.",  # 14
        "query: The Vinemount Ridge site in the Ontario Niagara Peninsula (approximately 43.19 N, -79.68 W) lies just above the brow of the Niagara Escarpment where a cool-climate, lake-moderated region with early spring warming, warm summers and moderately cool nights produces a relatively high diurnal range, and is underlain by deep, well-drained clay-loam and silty soils enriched with shale and natural moraine drainage that balance moisture retention with good aeration for balanced vine growth and fruit development.",  # 15
        "query: The Creek Shores site in the Ontario Niagara Peninsula (approximately 43.18 N, -79.33 W) lies on the Lake Iroquois Plain with a cool, moderating Lake Ontario influence that brings gradual spring warming, warm summers with cool nights, and an extended growing season, and is underlain by diverse, well-drained glacial delta sediments and sedimentary soils including lighter sands and porous loams over Queenston Shale that support balanced vine growth and concentrated fruit.",  # 16
        "query: The Four Mile Creek site in the Ontario Niagara Peninsula (approximately 43.25 N, -79.12 W) lies on a relatively flat, inland plain with a cool-climate, lake-moderated growing season featuring warm, sunny days and cool nights that extend ripening and preserve acidity, and is underlain by fertile, well-drained soils of glacial sediments over Queenston shale, including rich clay loams that retain moisture yet provide good drainage for balanced vine growth and fruit development.",  # 17
        "query: The Niagara Lakeshore site in the Ontario Niagara Peninsula (approximately 43.01 N, -79.08 W) sits along the southern shore of Lake Ontario with a cool, lake-moderated climate that features a long, consistent growing season with warm days, cool nights, and extended ripening into late October, and is underlain by well-drained glacial and lake-deposited soils of clay, silt, deltaic sands, and reddish sandy soils near the shore that support deep rooting and balanced moisture for quality vine growth.",  # 18
        "query: The Niagara River site in the Ontario Niagara Peninsula (approximately 43.90 N, -79.80 W) occupies a narrow strip of east-facing, gently sloped vineyards along the Niagara River with a cool-climate, lake- and river-moderated growing season of warm days and cool nights that extends ripening and mitigates frost risk, and is underlain by predominantly well-drained stratified fine sands over red Queenston shale with variable drainage that encourages deep root penetration and balanced vine growth.",  # 19
        "query: The Short Hills Bench site in the Ontario Niagara Peninsula (approximately 43.10 N, -79.28 W) has a cool-climate, lake-moderated growing season with warm, sunny days and cool nights that extend ripening and promote acidity, and is underlain by deep, varied soils of water-stratified clay, brown silty clay and sand/gravels with good drainage and water-holding capacity that support balanced vine growth and concentrated fruit character.",  # 20
        "query: The St. David's Bench site in the Ontario Niagara Peninsula (approximately 43.158 N, -79.124 W) lies on a gently sloping bench south of Lake Ontario with a cool-climate, lake- and escarpment-influenced growing season marked by early spring warming, warm sunny days and cool nights that extend ripening, and is underlain by deep, moisture-retentive silty clay and clay-loam soils over red sandstone that drain moderately and support balanced vine growth and concentrated fruit.",  # 21
        "query: The Prosser viticultural area in Washington, United States (approximately 46.2189 N, -119.7025 W) is situated in the heart of the Yakima Valley AVA where a semi-arid, high-sunshine climate with hot, dry summers, cool nights, and low annual precipitation supports a long growing season under irrigation, and the region's soils are typically well-drained silt-loams and gravelly alluvial deposits over basalt bedrock that favor deep rooting and balanced vine growth for premium wine grapes.",  # 22
        "query: The Michigan site in the Lake Michigan Shore region of the United States (approximately 42.60 N, -86.10 W) lies along the eastern shore of Lake Michigan where a cool, lake-moderated continental climate brings warm summer days, cool nights, and a relatively long growing season, and the vineyards are underlain by well-drained glacial sandy loams and gravelly soils that provide balanced drainage and moderate fertility for steady vine growth and fruit development.",
        "query: The Beaver site in Nova Scotia, Canada (approximately 45.10 N, -64.50 W) lies in a cool maritime region influenced by the Bay of Fundy where mild summers, cool nights, and a gradual autumn create a moderate growing season, and the vineyards are underlain by well-drained glacial and alluvial soils of sandy loams and gravels that support balanced moisture and resilient vine growth with moderate fertility and favorable heat retention",
        "query: The Bejo site in New York State, United States (approximately 42.75 N, -76.10 W) lies within the broader Finger Lakes region where deep glacial lakes moderate a cool continental climate characterized by warm summer days, cool nights, and an extended autumn ripening period, and the vineyards are underlain by well-drained glacial till soils consisting of gravelly loams, silts, and shale-derived sediments that allow deep root penetration and balanced water availability for consistent vine growth and fruit maturation.",
        "query: The Geneva site in New York State, United States (approximately 42.88 N, -76.98 W) sits along the northern shores of Seneca Lake in the Finger Lakes region where deep lake waters moderate winter cold and extend the growing season with warm days and cool nights, and the surrounding vineyards are typically planted on well-drained glacial till soils composed of silty loams, gravel deposits, and shale fragments over sedimentary bedrock that support deep rooting, steady moisture availability, and balanced vine growth.",
        "query: The Hudson site in New York State, United States (approximately 42.25 N, -73.79 W) lies in the Hudson Valley along the Hudson River where a cool continental climate moderated by the river brings warm summer days, cool nights, and a moderate growing season, and the vineyards are underlain by well-drained glacial and alluvial soils of silt loam, sand, and gravel over shale and limestone that support balanced vine growth and steady fruit development.",
        "query: The Portland site in New York State, United States (approximately 42.38 N, -79.47 W) lies along the southern shore of Lake Erie within the Lake Erie viticultural region where a cool, lake-moderated climate brings warm days, cool nights, and a relatively long growing season, and the vineyards are underlain by well-drained glacial silts, clays, and gravelly loams that retain moisture while supporting consistent vine growth and fruit ripening.",
        "query: The Montreal site in Quebec, Canada (approximately 45.50 N, -73.57 W) lies in a cool continental climate strongly influenced by the St. Lawrence River, with warm, humid summers, cool nights, and a relatively short but concentrated growing season marked by moderate frost risk, and the vineyards are underlain by well-drained glacial and alluvial soils of sandy loams, silts, and clay-rich deposits over sedimentary limestone bedrock that support balanced moisture retention and moderate vine vigor for cold-hardy varieties.",
        "query: The Madison site in Wisconsin, United States (approximately 43.07 N, -89.40 W) lies in a cool continental climate moderated by nearby lakes Mendota and Monona, with warm, sunny summers, pronounced cool nights, and a moderate growing season with meaningful frost risk, and the vineyards are underlain by well-drained glacial till and outwash soils of sandy loams, gravels, and silt-rich deposits over dolomitic limestone bedrock that encourage deep rooting and balanced drainage for cold-hardy and hybrid vine varieties.",
        "query: The Brownfield site on the Texas High Plains, United States (approximately 33.18 N, -102.27 W) lies at high elevation on the Llano Estacado plateau in a semi-arid continental climate with intense sunshine, hot days, strong diurnal temperature swings, and very low annual precipitation that require irrigation while naturally concentrating fruit flavors and preserving acidity, and the vineyards are underlain by well-drained, calcareous sandy loams that promote deep rooting, low vigor, and concentrated berry development across a wide range of vinifera varieties.",
        "query: The Penn State site in central Pennsylvania, United States (approximately 40.79 N, -77.86 W) lies in the Ridge and Valley physiographic region with a cool to moderate continental climate featuring warm, humid summers, cool nights, and a moderate growing season with moderate frost risk and meaningful rainfall, and the vineyards are underlain by well-drained residual and colluvial soils of shale, sandstone, and limestone-derived loams and silt-clay mixtures that offer varied drainage and moderate fertility supporting balanced vine growth.",
    ]
)

SITE_STR = np.array(
    [
        "query: WestKelowna",  # 0
        "query: OKFallsEast",  # 1
        "query: OKFallsWest",  # 2
        "query: OliverEast",  # 3
        "query: OliverSouth",  # 4
        "query: OliverWest",  # 5
        "query: BlackSage",  # 6
        "query: OsoyoosNorth",  # 7
        "query: OsoyoosNortheast",  # 8
        "query: OsoyoosSoutheast",  # 9
        "query: OsoyoosWest",  # 10
        "query: NaramataBench",  # 11
        "query: BeamsvilleBench",  # 12
        "query: LincolnLakeshore",  # 13
        "query: TwentyMileBench",  # 14
        "query: VinemountRidge",  # 15
        "query: CreekShores",  # 16
        "query: FourMileCreek",  # 17
        "query: NiagaraLakeshores",  # 18
        "query: NiagaraRiver",  # 19
        "query: ShortHillsBench",  # 20
        "query: StDavidsBench",  # 21
        "query: Prosser",  # 22
        "query: Mich1",
        "query: Beaver",
        "query: Bejo",
        "query: Geneva",
        "query: Hudson",
        "query: Portland",
        "query: Montreal",
        "query: Madison",
        "query: Brownfield",
        "query: PennState",
    ]
)

SITE_STR_MINI = np.array(
    [
        "query: West Kelowna, BC is a semi-arid, rain-shadow site with hot summers, cold winters, and well-drained volcanic and glacial soils.",
        "query: Okanagan Falls East, BC is a semi-arid, lake-moderated site with warm days, cool nights, and well-drained glaciofluvial gravel and sandy loam soils.",
        "query: Okanagan Falls West, BC is a warm, semi-arid site with hot days, cool nights, and well-drained glaciofluvial gravel, sand, and loam soils.",
        "query: Oliver East, BC is a warm, semi-arid continental site with hot summers, strong diurnal shifts, and well-drained sandy and gravelly glaciofluvial soils.",
        "query: Oliver South, BC is a hot, semi-arid desert-influenced site with cool nights and well-drained glaciofluvial sandy-gravel soils with low moisture retention.",
        "query: Oliver West, BC is a warm, semi-arid continental site with hot, dry summers, cool nights, and well-drained glaciofluvial gravels and sandy loams.",
        "query: Black Sage Bench, BC is a very warm semi-arid pocket desert site with hot summers, cool nights, and deep well-drained sandy glaciofluvial soils.",
        "query: Osoyoos North, BC is a hot, semi-arid desert-influenced site with cool nights and deep, well-drained sandy and glaciofluvial soils.",
        "query: Osoyoos North East, BC is a hot, semi-arid rain-shadow site with cool nights and well-drained glaciofluvial sands, gravels, and mineral-rich loams.",
        "query: Osoyoos South East Bench, BC is a hot, semi-arid rain-shadow site with long dry summers, cool nights, and well-drained sandy glaciofluvial soils with gravel.",
        "query: Osoyoos West, BC is a hot, semi-arid continental site with cool nights and well-drained sandy glaciofluvial soils with gravel and loam.",
        "query: Naramata Bench, BC is a moderate continental site lake-moderated by Okanagan Lake with warm summers, cool nights, and well-drained sandy to silt-loam glacial soils.",
        "query: Beamsville Bench, ON is a cool lake-moderated site with warm sunny days, Lake Ontario breezes, and deep well-drained heterogeneous soils over dolomitic limestone.",
        "query: Lincoln Lakeshore, ON is a cool lake-moderated site with a long temperate growing season and well-drained variable sandy soils and red clay loam.",
        "query: Twenty Mile Bench, ON is a cool lake-moderated site with escarpment shelter and deep, well-drained glacial clay and till soils rich in limestone and shale.",
        "query: Vinemount Ridge, ON is a cool lake-moderated escarpment site with high diurnal range and deep, well-drained clay-loam and silty soils with shale and moraine drainage.",
        "query: Creek Shores, ON is a cool lake-moderated site on the Lake Iroquois Plain with warm summers, extended growing season, and well-drained glacial sands and porous loams.",
        "query: Four Mile Creek, ON is a cool lake-moderated inland site with warm sunny days, cool nights, and fertile, well-drained clay loams over Queenston shale.",
        "query: Niagara Lakeshore, ON is a cool lake-moderated site with a long growing season and well-drained glacial and lake-deposited clay, silt, and deltaic sands.",
        "query: Niagara River, ON is a cool river- and lake-moderated site with warm days, cool nights, and well-drained stratified fine sands over red Queenston shale.",
        "query: Short Hills Bench, ON is a cool lake-moderated site with warm sunny days, cool nights, and deep, varied water-stratified clay, silty clay, and gravelly soils.",
        "query: St. David's Bench, ON is a cool lake- and escarpment-influenced site with warm days, cool nights, and deep silty clay and clay-loam soils over red sandstone.",
        "query: Prosser, WA is a semi-arid high-sunshine Yakima Valley site with hot, dry summers, cool nights, and well-drained silt-loams and gravelly alluvial soils over basalt.",
        "query: Michigan Lake Shore is a cool lake-moderated continental site with warm summers, cool nights, and well-drained glacial sandy loams and gravelly soils.",
        "query: Beaver, NS is a cool maritime site influenced by the Bay of Fundy with mild summers, cool nights, and well-drained glacial sandy loams and gravels.",
        "query: Bejo, NY is a cool continental Finger Lakes site with warm summers, cool nights, and well-drained glacial till of gravelly loams and shale-derived sediments.",
        "query: Geneva, NY is a cool Finger Lakes site lake-moderated by Seneca Lake with warm days, cool nights, and well-drained glacial till of silty loams and shale over sedimentary bedrock.",
        "query: Hudson, NY is a cool river-moderated Hudson Valley site with warm summers, cool nights, and well-drained glacial and alluvial silt loam and gravel over shale and limestone.",
        "query: Portland, NY is a cool lake-moderated Lake Erie site with warm days, cool nights, and well-drained glacial silts, clays, and gravelly loams.",
        "query: Montreal, QC is a cool continental site influenced by the St. Lawrence River with warm humid summers and well-drained glacial sandy loams and clay over limestone.",
        "query: Madison, WI is a cool continental lake-moderated site with warm summers, cool nights, and well-drained glacial sandy loams and gravels over dolomitic limestone.",
        "query: Brownfield, TX is a high-elevation semi-arid High Plains site with intense sun, hot days, strong diurnal swings, and well-drained calcareous sandy loams.",
        "query: Penn State, PA is a cool to moderate continental site with warm humid summers, cool nights, and well-drained shale, sandstone, and limestone-derived loams.",
    ]
)

SITE_STR_BARE = np.array(
    [
        "query: West Kelowna, BC; semi-arid, rain-shadow; hot summers, cold winters; volcanic and glacial soils.",
        "query: Okanagan Falls East, BC; semi-arid, lake-moderated; warm days, cool nights; glaciofluvial gravel and sandy loam.",
        "query: Okanagan Falls West, BC; semi-arid; hot days, cool nights; glaciofluvial gravel, sand, loam.",
        "query: Oliver East, BC; semi-arid continental; hot summers, strong diurnal shifts; sandy gravelly glaciofluvial soils.",
        "query: Oliver South, BC; hot semi-arid desert-influenced; cool nights; sandy-gravel soils, low moisture retention.",
        "query: Oliver West, BC; semi-arid continental; hot dry summers, cool nights; glaciofluvial gravels and sandy loams.",
        "query: Black Sage Bench, BC; hot semi-arid pocket desert; cool nights; deep sandy glaciofluvial soils.",
        "query: Osoyoos North, BC; hot semi-arid desert-influenced; cool nights; deep sandy glaciofluvial soils.",
        "query: Osoyoos North East, BC; hot semi-arid rain-shadow; cool nights; glaciofluvial sands, gravels, mineral loams.",
        "query: Osoyoos South East Bench, BC; hot semi-arid rain-shadow; long dry summers, cool nights; sandy glaciofluvial soils with gravel.",
        "query: Osoyoos West, BC; hot semi-arid continental; cool nights; sandy glaciofluvial soils with gravel and loam.",
        "query: Naramata Bench, BC; moderate continental, lake-moderated; warm summers, cool nights; sandy to silt-loam glacial soils.",
        "query: Beamsville Bench, ON; cool, lake-moderated; warm days, Lake Ontario breezes; heterogeneous soils over dolomitic limestone.",
        "query: Lincoln Lakeshore, ON; cool, lake-moderated; long temperate season; sandy soils and red clay loam.",
        "query: Twenty Mile Bench, ON; cool, lake-moderated, escarpment shelter; glacial clay and till over limestone and shale.",
        "query: Vinemount Ridge, ON; cool, lake-moderated escarpment; high diurnal range; clay-loam and silty soils with shale.",
        "query: Creek Shores, ON; cool, lake-moderated; warm summers, extended season; glacial sands and porous loams.",
        "query: Four Mile Creek, ON; cool, lake-moderated inland; warm days, cool nights; clay loams over Queenston shale.",
        "query: Niagara Lakeshore, ON; cool, lake-moderated; long growing season; glacial and lake-deposited clay, silt, deltaic sands.",
        "query: Niagara River, ON; cool, river- and lake-moderated; warm days, cool nights; fine sands over red Queenston shale.",
        "query: Short Hills Bench, ON; cool, lake-moderated; warm days, cool nights; water-stratified clay, silty clay, gravels.",
        "query: St. David's Bench, ON; cool, lake- and escarpment-influenced; warm days, cool nights; silty clay and clay-loam over red sandstone.",
        "query: Prosser, WA; semi-arid, high-sunshine Yakima Valley; hot dry summers, cool nights; silt-loams and gravelly alluvial soils over basalt.",
        "query: Michigan Lake Shore; cool, lake-moderated continental; warm summers, cool nights; glacial sandy loams and gravels.",
        "query: Beaver, NS; cool maritime, Bay of Fundy influence; mild summers, cool nights; glacial sandy loams and gravels.",
        "query: Bejo, NY; cool continental, Finger Lakes; warm summers, cool nights; glacial till of gravelly loams and shale sediments.",
        "query: Geneva, NY; cool Finger Lakes, Seneca Lake-moderated; warm days, cool nights; glacial silty loams and shale over sedimentary bedrock.",
        "query: Hudson, NY; cool, river-moderated Hudson Valley; warm summers, cool nights; glacial silt loam and gravel over shale and limestone.",
        "query: Portland, NY; cool, lake-moderated Lake Erie; warm days, cool nights; glacial silts, clays, and gravelly loams.",
        "query: Montreal, QC; cool continental, St. Lawrence influence; warm humid summers, short season; glacial sandy loams and clay over limestone.",
        "query: Madison, WI; cool continental, lake-moderated; warm summers, cool nights; glacial sandy loams and gravels over dolomitic limestone.",
        "query: Brownfield, TX; high-elevation semi-arid High Plains; intense sun, hot days, strong diurnal swings; calcareous sandy loams.",
        "query: Penn State, PA; cool to moderate continental; warm humid summers, cool nights; shale, sandstone, and limestone-derived loams.",
    ]
)

STATION_STR_VERB = np.array(
    [
        "query: The Kelowna weather station in the Okanagan Valley of British Columbia (49.96 N, -119.30 W) records a continental, semi-arid climate with dry, hot, sunny summers and cool, relatively dry winters moderated by nearby Okanagan Lake, yielding long growing seasons with warm days and cool nights and modest annual precipitation.",
        "query: The Kelowna UBCO weather station in the Okanagan Valley of British Columbia (49.9408 N, -119.4002 W) records a semi-arid, continental climate with hot, dry, sunny summers and cool, relatively dry winters moderated locally by the valley and nearby Okanagan Lake, yielding long growing seasons with warm days, cool nights, low precipitation, and notable diurnal temperature variation.",
        "query: The Osoyoos CS weather station in the Okanagan Valley of British Columbia (49.03 N, -119.44 W) records a cold, semi-arid climate with hot, dry summers, cool nights, and relatively mild winters under a strong rain shadow that produces low annual precipitation and long growing seasons",
        "query: The Summerland weather station in the Okanagan Valley of British Columbia (49.57 N, -119.65 W) records a semi-arid continental climate with hot, dry, sunny summers and cool, relatively mild winters moderated by nearby Okanagan Lake, producing one of the warmest and driest growing seasons in Canada with low annual precipitation and strong diurnal temperature swings.",
        "query: The Grimsby Mountain weather station on the Ontario Niagara Peninsula (43.187 N, -79.565 W) records a humid continental climate with warm, sometimes hot summers and cold, snowy winters influenced by nearby Lake Ontario and the Niagara Escarpment that extend the growing season and moderate extreme temperatures.",
        "query: The Vineland Station weather stati"
        "on on the Ontario Niagara Peninsula (43.18 N, -79.40 W) records a humid continental climate with warm, humid summers, cold winters, and significant year-round precipitation moderated by nearby Lake Ontario and local topography.",
        "query: The Roza2 weather station in Washington, United States is associated with the broader Roza/Prosser area in south-central Washington (46.78 N, -120.45 W) and records a semi-arid, continental climate with hot, dry summers, cool nights, and low annual precipitation typical of the Yakima Valley region that supports irrigated agriculture and long growing seasons",
        "query: The SWMREC weather station in southwest Michigan, United States (approximately 42.09 N, -86.36 W) records a humid continental climate with warm, humid summers and cold winters moderated by nearby Lake Michigan, producing relatively long growing seasons with warm days, cool nights, and moderate annual precipitation typical of the Lake Michigan Shore region.",
        "query: The Beaver weather station in Nova Scotia, Canada (approximately 45.10 N, -64.50 W) records a cool maritime climate influenced by the Bay of Fundy and the Atlantic Ocean with mild summers, cool nights, and relatively moderate winters, producing a moderate growing season with frequent moisture and modest annual precipitation.",
        "query: The Bejo weather station in the Finger Lakes region of New York State, United States (approximately 42.75 N, -76.10 W) records a humid continental climate with warm summers, cool nights, and cold winters moderated locally by nearby Finger Lakes, producing a moderate growing season with balanced precipitation and notable seasonal temperature variation.",
        "query: The Geneva weather station in the Finger Lakes region of New York State, United States (approximately 42.87 N, -76.99 W) records a humid continental climate with warm summers, cool nights, and cold winters moderated by nearby Seneca Lake, producing a relatively long growing season with steady precipitation and significant diurnal temperature variation.",
        "query: The Hudson weather station in the Hudson Valley of New York State, United States (approximately 42.25 N, -73.79 W) records a humid continental climate with warm, humid summers and cold winters influenced by the Hudson River valley, producing a moderate growing season with consistent precipitation and seasonal temperature variability.",
        "query: The Portland weather station along the Lake Erie shoreline of New York State, United States (approximately 42.38 N, -79.47 W) records a humid continental climate with warm summers, cool nights, and cold winters moderated by nearby Lake Erie, producing a relatively long growing season with lake-influenced temperatures and moderate annual precipitation.",
        "query: The Montreal weather station in Quebec, Canada (approximately 45.47 N, -73.74 W) records a humid continental climate with warm, humid summers, cool nights, and cold, snowy winters influenced by the St. Lawrence River valley, producing a moderate growing season with consistent precipitation distributed across the year and notable seasonal temperature variability.",
        "query: The Madison weather station in Wisconsin, United States (approximately 43.14 N, -89.33 W) records a humid continental climate with warm, humid summers, cool nights, and cold, snowy winters moderated modestly by the surrounding lakes Mendota and Monona, producing a moderate growing season with fairly consistent precipitation and significant diurnal and seasonal temperature variation.",
        "query: The Brownfield weather station on the Texas High Plains, United States (approximately 33.18 N, -102.27 W) records a semi-arid continental climate with hot, intensely sunny summers, strong diurnal temperature swings, cool nights, and low annual precipitation typical of the Llano Estacado plateau that produces a long growing season with high evapotranspiration demand and occasional late spring or early autumn frost risk.",
        "query: The Penn State weather station in central Pennsylvania, United States (approximately 40.72 N, -77.93 W) records a humid continental climate with warm, humid summers, cool nights, and cold winters influenced by the Ridge and Valley terrain that channels air flow and moderates temperature extremes, producing a moderate growing season with well-distributed annual precipitation and meaningful diurnal temperature swings.",
    ]
)

STATION_STR_MINI = np.array(
    [
        "query: Kelowna, BC is a continental semi-arid station with hot, dry summers and cool winters moderated by Okanagan Lake, with long growing seasons and low precipitation.",
        "query: Kelowna UBCO, BC is a semi-arid continental station with hot, dry summers, cool winters, low precipitation, and notable diurnal variation moderated by the valley and Okanagan Lake.",
        "query: Osoyoos CS, BC is a semi-arid rain-shadow station with hot, dry summers, cool nights, mild winters, low annual precipitation, and long growing seasons.",
        "query: Summerland, BC is a semi-arid continental station with hot, dry summers, cool mild winters moderated by Okanagan Lake, low precipitation, and strong diurnal swings.",
        "query: Grimsby Mountain, ON is a humid continental station with warm summers and cold snowy winters moderated by Lake Ontario and the Niagara Escarpment, with an extended growing season.",
        "query: Vineland Station, ON is a humid continental station with warm humid summers, cold winters, and significant year-round precipitation moderated by Lake Ontario.",
        "query: Roza2, WA is a semi-arid continental station with hot, dry summers, cool nights, and low annual precipitation typical of the Yakima Valley with long growing seasons.",
        "query: SWMREC, MI is a humid continental station with warm humid summers and cold winters moderated by Lake Michigan, with long growing seasons and moderate precipitation.",
        "query: Beaver, NS is a cool maritime station influenced by the Bay of Fundy with mild summers, cool nights, moderate winters, and frequent moisture.",
        "query: Bejo, NY is a humid continental Finger Lakes station with warm summers, cool nights, cold winters moderated by nearby lakes, and balanced precipitation.",
        "query: Geneva, NY is a humid continental Finger Lakes station with warm summers, cool nights, cold winters moderated by Seneca Lake, and a long growing season with steady precipitation.",
        "query: Hudson, NY is a humid continental station with warm humid summers, cold winters influenced by the Hudson River valley, and consistent precipitation.",
        "query: Portland, NY is a humid continental station with warm summers, cool nights, and cold winters moderated by Lake Erie, with a long growing season and moderate precipitation.",
        "query: Montreal, QC is a humid continental station with warm humid summers, cold snowy winters influenced by the St. Lawrence River valley, and consistent year-round precipitation.",
        "query: Madison, WI is a humid continental station with warm humid summers, cold snowy winters modestly moderated by surrounding lakes, and consistent precipitation with strong diurnal variation.",
        "query: Brownfield, TX is a semi-arid continental High Plains station with hot, sunny summers, strong diurnal swings, cool nights, low precipitation, and a long growing season.",
        "query: Penn State, PA is a humid continental station with warm humid summers, cold winters influenced by Ridge and Valley terrain, moderate growing season, and well-distributed precipitation.",
    ]
)

STATION_STR_BARE = np.array(
    [
        "query: Kelowna, BC; semi-arid continental; hot dry summers, cool winters; Okanagan Lake moderated; low precipitation.",
        "query: Kelowna UBCO, BC; semi-arid continental; hot dry summers, cool winters; low precipitation; strong diurnal variation.",
        "query: Osoyoos CS, BC; semi-arid, rain-shadow; hot dry summers, cool nights, mild winters; low precipitation; long growing season.",
        "query: Summerland, BC; semi-arid continental; hot dry summers, cool mild winters; Okanagan Lake moderated; strong diurnal swings.",
        "query: Grimsby Mountain, ON; humid continental; warm summers, cold snowy winters; Lake Ontario and escarpment moderated; extended growing season.",
        "query: Vineland Station, ON; humid continental; warm humid summers, cold winters; Lake Ontario moderated; significant year-round precipitation.",
        "query: Roza2, WA; semi-arid continental, Yakima Valley; hot dry summers, cool nights; low precipitation; long growing season.",
        "query: SWMREC, MI; humid continental; warm humid summers, cold winters; Lake Michigan moderated; moderate precipitation; long growing season.",
        "query: Beaver, NS; cool maritime, Bay of Fundy; mild summers, cool nights, moderate winters; frequent moisture.",
        "query: Bejo, NY; humid continental, Finger Lakes; warm summers, cool nights, cold winters; balanced precipitation.",
        "query: Geneva, NY; humid continental, Finger Lakes; warm summers, cool nights, cold winters; Seneca Lake moderated; long growing season.",
        "query: Hudson, NY; humid continental, Hudson River Valley; warm humid summers, cold winters; consistent precipitation.",
        "query: Portland, NY; humid continental, Lake Erie; warm summers, cool nights, cold winters; moderate precipitation; long growing season.",
        "query: Montreal, QC; humid continental, St. Lawrence Valley; warm humid summers, cold snowy winters; consistent year-round precipitation.",
        "query: Madison, WI; humid continental; warm humid summers, cold snowy winters; lake-moderated; consistent precipitation; strong diurnal variation.",
        "query: Brownfield, TX; semi-arid continental, High Plains; hot sunny summers, cool nights; strong diurnal swings; low precipitation; long growing season.",
        "query: Penn State, PA; humid continental, Ridge and Valley; warm humid summers, cold winters; well-distributed precipitation.",
    ]
)

REGION_STR_VERB = np.array(
    [
        "query: The Okanagan Valley in British Columbia is a semi-arid, continental-influenced wine region with long, warm, sunny summers, cool nights, and low annual precipitation, where well-drained soils and pronounced diurnal temperature swings create ideal conditions for ripening a wide range of vinifera grapes while preserving acidity and flavor concentration.",
        "query: The Ontario Niagara Peninsula is a cool-climate, lake-moderated wine region with warm, sunny summers, cool nights, and fertile, well-drained soils over limestone and glacial deposits, creating an extended growing season that supports balanced ripening and high-quality vinifera grape production.",
        "query: Washington State is a semi-arid, continental-influenced wine region with hot, sunny days, cool nights, and low annual precipitation in the rain-shadow of the Cascade Mountains, where well-drained alluvial and volcanic soils combined with irrigation allow precise control of vine vigor and high-quality vinifera grape production.",
        "query: New York State in the United States is a cool- to moderate-climate wine region strongly influenced by large lakes and river valleys, with warm summers, cool nights, and cold winters moderated locally by the Finger Lakes, Lake Erie, and Lake Ontario, where glacially derived soils of shale, silt, and gravel support balanced vine growth and high-quality vinifera grape production.",
        "query: Nova Scotia in Canada is a cool maritime wine region influenced by the Atlantic Ocean and the Bay of Fundy, with mild summers, cool nights, and relatively moderate winters, where well-drained glacial and coastal soils and a long, cool growing season support the production of fresh, aromatic vinifera and hybrid grape varieties.",
        "query: Michigan in the United States is a cool-climate, lake-moderated wine region surrounding Lake Michigan where warm summer days, cool nights, and long autumns extend the growing season, and well-drained glacial sandy loams and gravelly soils support balanced vine growth and high-quality vinifera grape production.",
        "query: Quebec in Canada is a cool to cold continental wine region moderated by the St. Lawrence River valley, with warm, humid summers, cool nights, and cold winters where well-drained glacial and alluvial soils of sandy loams and limestone-rich deposits support a short but concentrated growing season suited primarily to cold-hardy varieties with strong natural acidity and aromatic intensity.",
        "query: Wisconsin in the United States is a cool continental wine region moderated locally by Lakes Mendota, Michigan, and other inland waters, with warm, sunny summers, cool nights, and cold winters where well-drained soils of sandy loams, gravels, and dolomitic limestone-derived deposits support a moderate growing season suited to cold-hardy varieties with balanced acidity and fruit character.",
        "query: Texas in the United States is a warm to hot, semi-arid wine region with diverse growing areas ranging from the elevated, high-sunshine Texas High Plains to the Hill Country, where hot days, strong diurnal temperature swings, cool nights, and low annual precipitation combine with well-drained calcareous, sandy, and limestone-rich soils to support a long growing season.",
        "query: Pennsylvania in the United States is a moderate cool-climate wine region influenced by the Ridge and Valley terrain, the Susquehanna River, and proximity to Lake Erie in the northwest, with warm, humid summers, cool nights, and cold winters where well-drained soils support a moderate growing season suited to grape varieties.",
    ]
)

REGION_STR = np.array(
    [
        "query: British Colubmia Okanagan Valley",
        "query: Ontario Niagara Peninsula",
        "query: Washington, United States",
        "query: New York State, United States",
        "query: Nova Scotia, Canada",
        "query: Michigan, United States",
        "query: Montreal, Canada",
        "query: Wicsonsin State, United States",
        "query: Texas State, United States",
        "query: Pennsylvania State, United States",
    ]
)

REGION_STR_MINI = np.array(
    [
        "query: Okanagan Valley, BC is a semi-arid continental wine region with hot sunny summers, cool nights, low precipitation, and well-drained soils with strong diurnal swings favoring a wide range of vinifera.",
        "query: Ontario Niagara Peninsula is a cool, lake-moderated wine region with warm sunny summers, cool nights, and well-drained soils over limestone and glacial deposits supporting an extended growing season.",
        "query: Washington State is a semi-arid continental wine region with hot sunny days, cool nights, low precipitation, and well-drained alluvial and volcanic soils under irrigation supporting high-quality vinifera.",
        "query: New York State is a cool to moderate wine region lake-moderated by the Finger Lakes, Lake Erie, and Lake Ontario with warm summers, cool nights, and glacially derived shale, silt, and gravel soils.",
        "query: Nova Scotia is a cool maritime wine region influenced by the Atlantic Ocean and Bay of Fundy with mild summers, cool nights, and well-drained glacial soils supporting aromatic vinifera and hybrid varieties.",
        "query: Michigan is a cool, lake-moderated wine region with warm summer days, cool nights, long autumns, and well-drained glacial sandy loams and gravelly soils supporting high-quality vinifera.",
        "query: Quebec is a cool to cold continental wine region moderated by the St. Lawrence River with warm humid summers, cold winters, and glacial sandy loam and limestone soils suited to cold-hardy varieties.",
        "query: Wisconsin is a cool continental wine region with warm sunny summers, cool nights, cold winters, and well-drained sandy loams and dolomitic limestone soils suited to cold-hardy varieties.",
        "query: Texas is a warm to hot semi-arid wine region with hot days, strong diurnal swings, cool nights, low precipitation, and well-drained calcareous and limestone-rich soils supporting a long growing season.",
        "query: Pennsylvania is a moderate cool-climate wine region with warm humid summers, cool nights, cold winters, and well-drained soils influenced by Ridge and Valley terrain and proximity to Lake Erie.",
    ]
)

REGION_STR_BARE = np.array(
    [
        "query: Okanagan Valley, BC; semi-arid continental; hot sunny summers, cool nights; low precipitation; strong diurnal swings; wide vinifera range.",
        "query: Ontario Niagara Peninsula; cool, lake-moderated; warm sunny summers, cool nights; limestone and glacial soils; extended growing season.",
        "query: Washington State; semi-arid continental, Cascade rain-shadow; hot sunny days, cool nights; low precipitation; alluvial and volcanic soils; irrigated.",
        "query: New York State; cool to moderate, lake-moderated; warm summers, cool nights, cold winters; glacial shale, silt, and gravel soils.",
        "query: Nova Scotia; cool maritime, Atlantic and Bay of Fundy; mild summers, cool nights; glacial coastal soils; aromatic vinifera and hybrids.",
        "query: Michigan; cool, Lake Michigan-moderated; warm summers, cool nights, long autumns; glacial sandy loams and gravels.",
        "query: Quebec; cool to cold continental, St. Lawrence moderated; warm humid summers, cold winters; glacial sandy loams and limestone; cold-hardy varieties.",
        "query: Wisconsin; cool continental, lake-moderated; warm sunny summers, cold winters; sandy loams and dolomitic limestone; cold-hardy varieties.",
        "query: Texas; warm to hot semi-arid; hot days, cool nights, strong diurnal swings; low precipitation; calcareous and limestone soils; long growing season.",
        "query: Pennsylvania; moderate cool-climate, Ridge and Valley; warm humid summers, cool nights, cold winters; well-drained mixed soils.",
    ]
)
STATION_STR = np.array(
    [
        "query: Kelowna",
        "query: Kelowna UBC",
        "query: Osoyoos CS",
        "query: Summerland",
        "query: Grimbsy Mountain",
        "query: Vineland Station",
        "query: Roza2",
        "query: SWMREC",
        "query: Beaver",
        "query: Bejo",
        "query: Geneva",
        "query: Hudson",
        "query: Portland",
        "query: Montreal",
        "query: Madison",
        "query: Brownsfield",
        "query: PennState",
    ]
)

SITE_STR_TEST = np.array(
    [
        "query: a wax candle drips faster on its left side when tilted slightly.",
        "query: a silver spoon tarnishes more quickly in humid coastal air.",
        "query: a cotton pillowcase absorbs moisture more readily than a synthetic one.",
        "query: a wooden door swells in summer and shrinks back in winter.",
        "query: a brass doorbell button develops a greenish patina over decades of use.",
        "query: a wine cork left dry will eventually shrink and allow air into the bottle.",
        "query: a cast iron radiator takes longer to heat up but retains warmth for hours.",
        "query: a leather boot creases along the same fold lines with every step.",
        "query: a glass window pane expands very slightly on hot summer afternoons.",
        "query: a copper pipe develops a blue-green patina after years of water exposure.",
        "query: a wool sweater felts and shrinks if washed in hot water.",
        "query: a ceramic tile stays cool underfoot even in a warm room.",
        "query: a hand saw blade dulls faster when cutting green or wet wood.",
        "query: a rubber eraser hardens and becomes less effective after several years.",
        "query: a birch tree sheds its papery bark naturally each season.",
        "query: a steel nail rusts faster when exposed to saltwater than freshwater.",
        "query: a linen tablecloth softens considerably after a dozen wash cycles.",
        "query: a typewriter key sticks more often in cold and dry conditions.",
        "query: a magnifying glass can focus sunlight into a point hot enough to ignite paper.",
        "query: a wicker basket weakens at the weave points if left in standing water.",
        "query: a feather pillow loses loft gradually over years of nightly use.",
        "query: a terracotta pot absorbs water through its walls, keeping roots cooler.",
        "query: a clock spring stores more tension the further it is wound.",
        "query: a pencil eraser leaves pink residue on paper when new but less so when worn.",
        "query: a tin roof expands audibly with loud pops on warm sunny mornings.",
        "query: a horsehair brush holds paint longer than a synthetic bristle brush.",
        "query: a cotton wick burns more evenly when centered in a candle jar.",
        "query: a bamboo cutting board resists knife marks better than softwood boards.",
        "query: a wooden match ignites more reliably when struck at a shallow angle.",
        "query: a pearl necklace gains a deeper luster from the oils of human skin over time.",
        "query: a vinyl record warps permanently if left in direct sunlight.",
        "query: a stone mortar smooths out with repeated grinding use over the years.",
        "query: a silk scarf loses its sheen permanently if ironed at too high a temperature.",
    ]
)

STATION_STR_TEST = np.array(
    [
        "query: a barometer drops in pressure before a storm arrives.",
        "query: a sundial becomes useless on a cloudy or overcast day.",
        "query: a weather vane squeaks on its pivot when it has not been oiled.",
        "query: a rain gauge must be emptied after each recorded event for accuracy.",
        "query: a thermometer placed in direct sun reads several degrees too high.",
        "query: a hygrometer measures relative humidity using a strand of human hair.",
        "query: a lightning rod must be grounded properly to safely divert a strike.",
        "query: a wooden fence post rots fastest at the point where it meets the soil.",
        "query: a metal flagpole hums at a low frequency in strong sustained winds.",
        "query: a greenhouse traps infrared radiation even when outside temperatures drop sharply.",
        "query: a beehive loses heat rapidly if its entrance faces into a north wind.",
        "query: a stone wall absorbs heat during the day and radiates it slowly at night.",
        "query: a wind sock at an airfield goes fully horizontal in a strong gale.",
        "query: a frozen pipe is most likely to burst during the thaw rather than the freeze.",
        "query: a thatched roof provides surprisingly effective insulation in cold weather.",
        "query: a clay pot left outside will crack if water freezes inside its walls.",
        "query: a wooden shingle roof weathers to a silver-gray color after several seasons.",
    ]
)

REGION_STR_TEST = np.array(
    [
        "query: a river always carves its path along the line of least geological resistance.",
        "query: a valley floor collects cold air overnight as it drains from surrounding slopes.",
        "query: a coastal fog bank retreats inland each morning as the land warms.",
        "query: a mountain range casts a rain shadow on the leeward side of its peaks.",
        "query: a lake moderates nearby temperatures by absorbing heat in summer and releasing it in winter.",
        "query: a glacier grinds bedrock into fine powder called rock flour as it advances.",
        "query: a delta grows outward into the sea as a river deposits its sediment load.",
        "query: a limestone plateau dissolves slowly over centuries through rainwater erosion.",
        "query: a peninsula experiences milder winters than inland areas at the same latitude.",
        "query: a boreal forest floor remains frozen just below the surface even in midsummer.",
    ]
)
