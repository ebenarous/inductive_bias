from PIL import Image
from time import sleep
import numpy as np
import os
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import pandas as pd
import torchvision.datasets as datasets
from torchvision.datasets import CIFAR10, ImageNet, STL10, Cityscapes
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import torch.nn.functional as F

dict = {0: 'tench, Tinca tinca',
    1: 'goldfish, Carassius auratus',
    2: 'great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias',
    3: 'tiger shark, Galeocerdo cuvieri',
    4: 'hammerhead, hammerhead shark',
    5: 'electric ray, crampfish, numbfish, torpedo',
    6: 'stingray',
    7: 'cock',
    8: 'hen',
    9: 'ostrich, Struthio camelus',
    10: 'brambling, Fringilla montifringilla',
    11: 'goldfinch, Carduelis carduelis',
    12: 'house finch, linnet, Carpodacus mexicanus',
    13: 'junco, snowbird',
    14: 'indigo bunting, indigo finch, indigo bird, Passerina cyanea',
    15: 'robin, American robin, Turdus migratorius',
    16: 'bulbul',
    17: 'jay',
    18: 'magpie',
    19: 'chickadee',
    20: 'water ouzel, dipper',
    21: 'kite',
    22: 'bald eagle, American eagle, Haliaeetus leucocephalus',
    23: 'vulture',
    24: 'great grey owl, great gray owl, Strix nebulosa',
    25: 'European fire salamander, Salamandra salamandra',
    26: 'common newt, Triturus vulgaris',
    27: 'eft',
    28: 'spotted salamander, Ambystoma maculatum',
    29: 'axolotl, mud puppy, Ambystoma mexicanum',
    30: 'bullfrog, Rana catesbeiana',
    31: 'tree frog, tree-frog',
    32: 'tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui',
    33: 'loggerhead, loggerhead turtle, Caretta caretta',
    34: 'leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea',
    35: 'mud turtle',
    36: 'terrapin',
    37: 'box turtle, box tortoise',
    38: 'banded gecko',
    39: 'common iguana, iguana, Iguana iguana',
    40: 'American chameleon, anole, Anolis carolinensis',
    41: 'whiptail, whiptail lizard',
    42: 'agama',
    43: 'frilled lizard, Chlamydosaurus kingi',
    44: 'alligator lizard',
    45: 'Gila monster, Heloderma suspectum',
    46: 'green lizard, Lacerta viridis',
    47: 'African chameleon, Chamaeleo chamaeleon',
    48: 'Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis',
    49: 'African crocodile, Nile crocodile, Crocodylus niloticus',
    50: 'American alligator, Alligator mississipiensis',
    51: 'triceratops',
    52: 'thunder snake, worm snake, Carphophis amoenus',
    53: 'ringneck snake, ring-necked snake, ring snake',
    54: 'hognose snake, puff adder, sand viper',
    55: 'green snake, grass snake',
    56: 'king snake, kingsnake',
    57: 'garter snake, grass snake',
    58: 'water snake',
    59: 'vine snake',
    60: 'night snake, Hypsiglena torquata',
    61: 'boa constrictor, Constrictor constrictor',
    62: 'rock python, rock snake, Python sebae',
    63: 'Indian cobra, Naja naja',
    64: 'green mamba',
    65: 'sea snake',
    66: 'horned viper, cerastes, sand viper, horned asp, Cerastes cornutus',
    67: 'diamondback, diamondback rattlesnake, Crotalus adamanteus',
    68: 'sidewinder, horned rattlesnake, Crotalus cerastes',
    69: 'trilobite',
    70: 'harvestman, daddy longlegs, Phalangium opilio',
    71: 'scorpion',
    72: 'black and gold garden spider, Argiope aurantia',
    73: 'barn spider, Araneus cavaticus',
    74: 'garden spider, Aranea diademata',
    75: 'black widow, Latrodectus mactans',
    76: 'tarantula',
    77: 'wolf spider, hunting spider',
    78: 'tick',
    79: 'centipede',
    80: 'black grouse',
    81: 'ptarmigan',
    82: 'ruffed grouse, partridge, Bonasa umbellus',
    83: 'prairie chicken, prairie grouse, prairie fowl',
    84: 'peacock',
    85: 'quail',
    86: 'partridge',
    87: 'African grey, African gray, Psittacus erithacus',
    88: 'macaw',
    89: 'sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita',
    90: 'lorikeet',
    91: 'coucal',
    92: 'bee eater',
    93: 'hornbill',
    94: 'hummingbird',
    95: 'jacamar',
    96: 'toucan',
    97: 'drake',
    98: 'red-breasted merganser, Mergus serrator',
    99: 'goose',
    100: 'black swan, Cygnus atratus',
    101: 'tusker',
    102: 'echidna, spiny anteater, anteater',
    103: 'platypus, duckbill, duckbilled platypus, duck-billed platypus, Ornithorhynchus anatinus',
    104: 'wallaby, brush kangaroo',
    105: 'koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus',
    106: 'wombat',
    107: 'jellyfish',
    108: 'sea anemone, anemone',
    109: 'brain coral',
    110: 'flatworm, platyhelminth',
    111: 'nematode, nematode worm, roundworm',
    112: 'conch',
    113: 'snail',
    114: 'slug',
    115: 'sea slug, nudibranch',
    116: 'chiton, coat-of-mail shell, sea cradle, polyplacophore',
    117: 'chambered nautilus, pearly nautilus, nautilus',
    118: 'Dungeness crab, Cancer magister',
    119: 'rock crab, Cancer irroratus',
    120: 'fiddler crab',
    121: 'king crab, Alaska crab, Alaskan king crab, Alaska king crab, Paralithodes camtschatica',
    122: 'American lobster, Northern lobster, Maine lobster, Homarus americanus',
    123: 'spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish',
    124: 'crayfish, crawfish, crawdad, crawdaddy',
    125: 'hermit crab',
    126: 'isopod',
    127: 'white stork, Ciconia ciconia',
    128: 'black stork, Ciconia nigra',
    129: 'spoonbill',
    130: 'flamingo',
    131: 'little blue heron, Egretta caerulea',
    132: 'American egret, great white heron, Egretta albus',
    133: 'bittern',
    134: 'crane',
    135: 'limpkin, Aramus pictus',
    136: 'European gallinule, Porphyrio porphyrio',
    137: 'American coot, marsh hen, mud hen, water hen, Fulica americana',
    138: 'bustard',
    139: 'ruddy turnstone, Arenaria interpres',
    140: 'red-backed sandpiper, dunlin, Erolia alpina',
    141: 'redshank, Tringa totanus',
    142: 'dowitcher',
    143: 'oystercatcher, oyster catcher',
    144: 'pelican',
    145: 'king penguin, Aptenodytes patagonica',
    146: 'albatross, mollymawk',
    147: 'grey whale, gray whale, devilfish, Eschrichtius gibbosus, Eschrichtius robustus',
    148: 'killer whale, killer, orca, grampus, sea wolf, Orcinus orca',
    149: 'dugong, Dugong dugon',
    150: 'sea lion',
    151: 'Chihuahua',
    152: 'Japanese spaniel',
    153: 'Maltese dog, Maltese terrier, Maltese',
    154: 'Pekinese, Pekingese, Peke',
    155: 'Shih-Tzu',
    156: 'Blenheim spaniel',
    157: 'papillon',
    158: 'toy terrier',
    159: 'Rhodesian ridgeback',
    160: 'Afghan hound, Afghan',
    161: 'basset, basset hound',
    162: 'beagle',
    163: 'bloodhound, sleuthhound',
    164: 'bluetick',
    165: 'black-and-tan coonhound',
    166: 'Walker hound, Walker foxhound',
    167: 'English foxhound',
    168: 'redbone',
    169: 'borzoi, Russian wolfhound',
    170: 'Irish wolfhound',
    171: 'Italian greyhound',
    172: 'whippet',
    173: 'Ibizan hound, Ibizan Podenco',
    174: 'Norwegian elkhound, elkhound',
    175: 'otterhound, otter hound',
    176: 'Saluki, gazelle hound',
    177: 'Scottish deerhound, deerhound',
    178: 'Weimaraner',
    179: 'Staffordshire bullterrier, Staffordshire bull terrier',
    180: 'American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier',
    181: 'Bedlington terrier',
    182: 'Border terrier',
    183: 'Kerry blue terrier',
    184: 'Irish terrier',
    185: 'Norfolk terrier',
    186: 'Norwich terrier',
    187: 'Yorkshire terrier',
    188: 'wire-haired fox terrier',
    189: 'Lakeland terrier',
    190: 'Sealyham terrier, Sealyham',
    191: 'Airedale, Airedale terrier',
    192: 'cairn, cairn terrier',
    193: 'Australian terrier',
    194: 'Dandie Dinmont, Dandie Dinmont terrier',
    195: 'Boston bull, Boston terrier',
    196: 'miniature schnauzer',
    197: 'giant schnauzer',
    198: 'standard schnauzer',
    199: 'Scotch terrier, Scottish terrier, Scottie',
    200: 'Tibetan terrier, chrysanthemum dog',
    201: 'silky terrier, Sydney silky',
    202: 'soft-coated wheaten terrier',
    203: 'West Highland white terrier',
    204: 'Lhasa, Lhasa apso',
    205: 'flat-coated retriever',
    206: 'curly-coated retriever',
    207: 'golden retriever',
    208: 'Labrador retriever',
    209: 'Chesapeake Bay retriever',
    210: 'German short-haired pointer',
    211: 'vizsla, Hungarian pointer',
    212: 'English setter',
    213: 'Irish setter, red setter',
    214: 'Gordon setter',
    215: 'Brittany spaniel',
    216: 'clumber, clumber spaniel',
    217: 'English springer, English springer spaniel',
    218: 'Welsh springer spaniel',
    219: 'cocker spaniel, English cocker spaniel, cocker',
    220: 'Sussex spaniel',
    221: 'Irish water spaniel',
    222: 'kuvasz',
    223: 'schipperke',
    224: 'groenendael',
    225: 'malinois',
    226: 'briard',
    227: 'kelpie',
    228: 'komondor',
    229: 'Old English sheepdog, bobtail',
    230: 'Shetland sheepdog, Shetland sheep dog, Shetland',
    231: 'collie',
    232: 'Border collie',
    233: 'Bouvier des Flandres, Bouviers des Flandres',
    234: 'Rottweiler',
    235: 'German shepherd, German shepherd dog, German police dog, alsatian',
    236: 'Doberman, Doberman pinscher',
    237: 'miniature pinscher',
    238: 'Greater Swiss Mountain dog',
    239: 'Bernese mountain dog',
    240: 'Appenzeller',
    241: 'EntleBucher',
    242: 'boxer',
    243: 'bull mastiff',
    244: 'Tibetan mastiff',
    245: 'French bulldog',
    246: 'Great Dane',
    247: 'Saint Bernard, St Bernard',
    248: 'Eskimo dog, husky',
    249: 'malamute, malemute, Alaskan malamute',
    250: 'Siberian husky',
    251: 'dalmatian, coach dog, carriage dog',
    252: 'affenpinscher, monkey pinscher, monkey dog',
    253: 'basenji',
    254: 'pug, pug-dog',
    255: 'Leonberg',
    256: 'Newfoundland, Newfoundland dog',
    257: 'Great Pyrenees',
    258: 'Samoyed, Samoyede',
    259: 'Pomeranian',
    260: 'chow, chow chow',
    261: 'keeshond',
    262: 'Brabancon griffon',
    263: 'Pembroke, Pembroke Welsh corgi',
    264: 'Cardigan, Cardigan Welsh corgi',
    265: 'toy poodle',
    266: 'miniature poodle',
    267: 'standard poodle',
    268: 'Mexican hairless',
    269: 'timber wolf, grey wolf, gray wolf, Canis lupus',
    270: 'white wolf, Arctic wolf, Canis lupus tundrarum',
    271: 'red wolf, maned wolf, Canis rufus, Canis niger',
    272: 'coyote, prairie wolf, brush wolf, Canis latrans',
    273: 'dingo, warrigal, warragal, Canis dingo',
    274: 'dhole, Cuon alpinus',
    275: 'African hunting dog, hyena dog, Cape hunting dog, Lycaon pictus',
    276: 'hyena, hyaena',
    277: 'red fox, Vulpes vulpes',
    278: 'kit fox, Vulpes macrotis',
    279: 'Arctic fox, white fox, Alopex lagopus',
    280: 'grey fox, gray fox, Urocyon cinereoargenteus',
    281: 'tabby, tabby cat',
    282: 'tiger cat',
    283: 'Persian cat',
    284: 'Siamese cat, Siamese',
    285: 'Egyptian cat',
    286: 'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor',
    287: 'lynx, catamount',
    288: 'leopard, Panthera pardus',
    289: 'snow leopard, ounce, Panthera uncia',
    290: 'jaguar, panther, Panthera onca, Felis onca',
    291: 'lion, king of beasts, Panthera leo',
    292: 'tiger, Panthera tigris',
    293: 'cheetah, chetah, Acinonyx jubatus',
    294: 'brown bear, bruin, Ursus arctos',
    295: 'American black bear, black bear, Ursus americanus, Euarctos americanus',
    296: 'ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus',
    297: 'sloth bear, Melursus ursinus, Ursus ursinus',
    298: 'mongoose',
    299: 'meerkat, mierkat',
    300: 'tiger beetle',
    301: 'ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle',
    302: 'ground beetle, carabid beetle',
    303: 'long-horned beetle, longicorn, longicorn beetle',
    304: 'leaf beetle, chrysomelid',
    305: 'dung beetle',
    306: 'rhinoceros beetle',
    307: 'weevil',
    308: 'fly',
    309: 'bee',
    310: 'ant, emmet, pismire',
    311: 'grasshopper, hopper',
    312: 'cricket',
    313: 'walking stick, walkingstick, stick insect',
    314: 'cockroach, roach',
    315: 'mantis, mantid',
    316: 'cicada, cicala',
    317: 'leafhopper',
    318: 'lacewing, lacewing fly',
    319: "dragonfly, darning needle, devil's darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk",
    320: 'damselfly',
    321: 'admiral',
    322: 'ringlet, ringlet butterfly',
    323: 'monarch, monarch butterfly, milkweed butterfly, Danaus plexippus',
    324: 'cabbage butterfly',
    325: 'sulphur butterfly, sulfur butterfly',
    326: 'lycaenid, lycaenid butterfly',
    327: 'starfish, sea star',
    328: 'sea urchin',
    329: 'sea cucumber, holothurian',
    330: 'wood rabbit, cottontail, cottontail rabbit',
    331: 'hare',
    332: 'Angora, Angora rabbit',
    333: 'hamster',
    334: 'porcupine, hedgehog',
    335: 'fox squirrel, eastern fox squirrel, Sciurus niger',
    336: 'marmot',
    337: 'beaver',
    338: 'guinea pig, Cavia cobaya',
    339: 'sorrel',
    340: 'zebra',
    341: 'hog, pig, grunter, squealer, Sus scrofa',
    342: 'wild boar, boar, Sus scrofa',
    343: 'warthog',
    344: 'hippopotamus, hippo, river horse, Hippopotamus amphibius',
    345: 'ox',
    346: 'water buffalo, water ox, Asiatic buffalo, Bubalus bubalis',
    347: 'bison',
    348: 'ram, tup',
    349: 'bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain sheep, Ovis canadensis',
    350: 'ibex, Capra ibex',
    351: 'hartebeest',
    352: 'impala, Aepyceros melampus',
    353: 'gazelle',
    354: 'Arabian camel, dromedary, Camelus dromedarius',
    355: 'llama',
    356: 'weasel',
    357: 'mink',
    358: 'polecat, fitch, foulmart, foumart, Mustela putorius',
    359: 'black-footed ferret, ferret, Mustela nigripes',
    360: 'otter',
    361: 'skunk, polecat, wood pussy',
    362: 'badger',
    363: 'armadillo',
    364: 'three-toed sloth, ai, Bradypus tridactylus',
    365: 'orangutan, orang, orangutang, Pongo pygmaeus',
    366: 'gorilla, Gorilla gorilla',
    367: 'chimpanzee, chimp, Pan troglodytes',
    368: 'gibbon, Hylobates lar',
    369: 'siamang, Hylobates syndactylus, Symphalangus syndactylus',
    370: 'guenon, guenon monkey',
    371: 'patas, hussar monkey, Erythrocebus patas',
    372: 'baboon',
    373: 'macaque',
    374: 'langur',
    375: 'colobus, colobus monkey',
    376: 'proboscis monkey, Nasalis larvatus',
    377: 'marmoset',
    378: 'capuchin, ringtail, Cebus capucinus',
    379: 'howler monkey, howler',
    380: 'titi, titi monkey',
    381: 'spider monkey, Ateles geoffroyi',
    382: 'squirrel monkey, Saimiri sciureus',
    383: 'Madagascar cat, ring-tailed lemur, Lemur catta',
    384: 'indri, indris, Indri indri, Indri brevicaudatus',
    385: 'Indian elephant, Elephas maximus',
    386: 'African elephant, Loxodonta africana',
    387: 'lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens',
    388: 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca',
    389: 'barracouta, snoek',
    390: 'eel',
    391: 'coho, cohoe, coho salmon, blue jack, silver salmon, Oncorhynchus kisutch',
    392: 'rock beauty, Holocanthus tricolor',
    393: 'anemone fish',
    394: 'sturgeon',
    395: 'gar, garfish, garpike, billfish, Lepisosteus osseus',
    396: 'lionfish',
    397: 'puffer, pufferfish, blowfish, globefish',
    398: 'abacus',
    399: 'abaya',
    400: "academic gown, academic robe, judge's robe",
    401: 'accordion, piano accordion, squeeze box',
    402: 'acoustic guitar',
    403: 'aircraft carrier, carrier, flattop, attack aircraft carrier',
    404: 'airliner',
    405: 'airship, dirigible',
    406: 'altar',
    407: 'ambulance',
    408: 'amphibian, amphibious vehicle',
    409: 'analog clock',
    410: 'apiary, bee house',
    411: 'apron',
    412: 'ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin',
    413: 'assault rifle, assault gun',
    414: 'backpack, back pack, knapsack, packsack, rucksack, haversack',
    415: 'bakery, bakeshop, bakehouse',
    416: 'balance beam, beam',
    417: 'balloon',
    418: 'ballpoint, ballpoint pen, ballpen, Biro',
    419: 'Band Aid',
    420: 'banjo',
    421: 'bannister, banister, balustrade, balusters, handrail',
    422: 'barbell',
    423: 'barber chair',
    424: 'barbershop',
    425: 'barn',
    426: 'barometer',
    427: 'barrel, cask',
    428: 'barrow, garden cart, lawn cart, wheelbarrow',
    429: 'baseball',
    430: 'basketball',
    431: 'bassinet',
    432: 'bassoon',
    433: 'bathing cap, swimming cap',
    434: 'bath towel',
    435: 'bathtub, bathing tub, bath, tub',
    436: 'beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon',
    437: 'beacon, lighthouse, beacon light, pharos',
    438: 'beaker',
    439: 'bearskin, busby, shako',
    440: 'beer bottle',
    441: 'beer glass',
    442: 'bell cote, bell cot',
    443: 'bib',
    444: 'bicycle-built-for-two, tandem bicycle, tandem',
    445: 'bikini, two-piece',
    446: 'binder, ring-binder',
    447: 'binoculars, field glasses, opera glasses',
    448: 'birdhouse',
    449: 'boathouse',
    450: 'bobsled, bobsleigh, bob',
    451: 'bolo tie, bolo, bola tie, bola',
    452: 'bonnet, poke bonnet',
    453: 'bookcase',
    454: 'bookshop, bookstore, bookstall',
    455: 'bottlecap',
    456: 'bow',
    457: 'bow tie, bow-tie, bowtie',
    458: 'brass, memorial tablet, plaque',
    459: 'brassiere, bra, bandeau',
    460: 'breakwater, groin, groyne, mole, bulwark, seawall, jetty',
    461: 'breastplate, aegis, egis',
    462: 'broom',
    463: 'bucket, pail',
    464: 'buckle',
    465: 'bulletproof vest',
    466: 'bullet train, bullet',
    467: 'butcher shop, meat market',
    468: 'cab, hack, taxi, taxicab',
    469: 'caldron, cauldron',
    470: 'candle, taper, wax light',
    471: 'cannon',
    472: 'canoe',
    473: 'can opener, tin opener',
    474: 'cardigan',
    475: 'car mirror',
    476: 'carousel, carrousel, merry-go-round, roundabout, whirligig',
    477: "carpenter's kit, tool kit",
    478: 'carton',
    479: 'car wheel',
    480: 'cash machine, cash dispenser, automated teller machine, automatic teller machine, automated teller, automatic teller, ATM',
    481: 'cassette',
    482: 'cassette player',
    483: 'castle',
    484: 'catamaran',
    485: 'CD player',
    486: 'cello, violoncello',
    487: 'cellular telephone, cellular phone, cellphone, cell, mobile phone',
    488: 'chain',
    489: 'chainlink fence',
    490: 'chain mail, ring mail, mail, chain armor, chain armour, ring armor, ring armour',
    491: 'chain saw, chainsaw',
    492: 'chest',
    493: 'chiffonier, commode',
    494: 'chime, bell, gong',
    495: 'china cabinet, china closet',
    496: 'Christmas stocking',
    497: 'church, church building',
    498: 'cinema, movie theater, movie theatre, movie house, picture palace',
    499: 'cleaver, meat cleaver, chopper',
    500: 'cliff dwelling',
    501: 'cloak',
    502: 'clog, geta, patten, sabot',
    503: 'cocktail shaker',
    504: 'coffee mug',
    505: 'coffeepot',
    506: 'coil, spiral, volute, whorl, helix',
    507: 'combination lock',
    508: 'computer keyboard, keypad',
    509: 'confectionery, confectionary, candy store',
    510: 'container ship, containership, container vessel',
    511: 'convertible',
    512: 'corkscrew, bottle screw',
    513: 'cornet, horn, trumpet, trump',
    514: 'cowboy boot',
    515: 'cowboy hat, ten-gallon hat',
    516: 'cradle',
    517: 'crane',
    518: 'crash helmet',
    519: 'crate',
    520: 'crib, cot',
    521: 'Crock Pot',
    522: 'croquet ball',
    523: 'crutch',
    524: 'cuirass',
    525: 'dam, dike, dyke',
    526: 'desk',
    527: 'desktop computer',
    528: 'dial telephone, dial phone',
    529: 'diaper, nappy, napkin',
    530: 'digital clock',
    531: 'digital watch',
    532: 'dining table, board',
    533: 'dishrag, dishcloth',
    534: 'dishwasher, dish washer, dishwashing machine',
    535: 'disk brake, disc brake',
    536: 'dock, dockage, docking facility',
    537: 'dogsled, dog sled, dog sleigh',
    538: 'dome',
    539: 'doormat, welcome mat',
    540: 'drilling platform, offshore rig',
    541: 'drum, membranophone, tympan',
    542: 'drumstick',
    543: 'dumbbell',
    544: 'Dutch oven',
    545: 'electric fan, blower',
    546: 'electric guitar',
    547: 'electric locomotive',
    548: 'entertainment center',
    549: 'envelope',
    550: 'espresso maker',
    551: 'face powder',
    552: 'feather boa, boa',
    553: 'file, file cabinet, filing cabinet',
    554: 'fireboat',
    555: 'fire engine, fire truck',
    556: 'fire screen, fireguard',
    557: 'flagpole, flagstaff',
    558: 'flute, transverse flute',
    559: 'folding chair',
    560: 'football helmet',
    561: 'forklift',
    562: 'fountain',
    563: 'fountain pen',
    564: 'four-poster',
    565: 'freight car',
    566: 'French horn, horn',
    567: 'frying pan, frypan, skillet',
    568: 'fur coat',
    569: 'garbage truck, dustcart',
    570: 'gasmask, respirator, gas helmet',
    571: 'gas pump, gasoline pump, petrol pump, island dispenser',
    572: 'goblet',
    573: 'go-kart',
    574: 'golf ball',
    575: 'golfcart, golf cart',
    576: 'gondola',
    577: 'gong, tam-tam',
    578: 'gown',
    579: 'grand piano, grand',
    580: 'greenhouse, nursery, glasshouse',
    581: 'grille, radiator grille',
    582: 'grocery store, grocery, food market, market',
    583: 'guillotine',
    584: 'hair slide',
    585: 'hair spray',
    586: 'half track',
    587: 'hammer',
    588: 'hamper',
    589: 'hand blower, blow dryer, blow drier, hair dryer, hair drier',
    590: 'hand-held computer, hand-held microcomputer',
    591: 'handkerchief, hankie, hanky, hankey',
    592: 'hard disc, hard disk, fixed disk',
    593: 'harmonica, mouth organ, harp, mouth harp',
    594: 'harp',
    595: 'harvester, reaper',
    596: 'hatchet',
    597: 'holster',
    598: 'home theater, home theatre',
    599: 'honeycomb',
    600: 'hook, claw',
    601: 'hoopskirt, crinoline',
    602: 'horizontal bar, high bar',
    603: 'horse cart, horse-cart',
    604: 'hourglass',
    605: 'iPod',
    606: 'iron, smoothing iron',
    607: "jack-o'-lantern",
    608: 'jean, blue jean, denim',
    609: 'jeep, landrover',
    610: 'jersey, T-shirt, tee shirt',
    611: 'jigsaw puzzle',
    612: 'jinrikisha, ricksha, rickshaw',
    613: 'joystick',
    614: 'kimono',
    615: 'knee pad',
    616: 'knot',
    617: 'lab coat, laboratory coat',
    618: 'ladle',
    619: 'lampshade, lamp shade',
    620: 'laptop, laptop computer',
    621: 'lawn mower, mower',
    622: 'lens cap, lens cover',
    623: 'letter opener, paper knife, paperknife',
    624: 'library',
    625: 'lifeboat',
    626: 'lighter, light, igniter, ignitor',
    627: 'limousine, limo',
    628: 'liner, ocean liner',
    629: 'lipstick, lip rouge',
    630: 'Loafer',
    631: 'lotion',
    632: 'loudspeaker, speaker, speaker unit, loudspeaker system, speaker system',
    633: "loupe, jeweler's loupe",
    634: 'lumbermill, sawmill',
    635: 'magnetic compass',
    636: 'mailbag, postbag',
    637: 'mailbox, letter box',
    638: 'maillot',
    639: 'maillot, tank suit',
    640: 'manhole cover',
    641: 'maraca',
    642: 'marimba, xylophone',
    643: 'mask',
    644: 'matchstick',
    645: 'maypole',
    646: 'maze, labyrinth',
    647: 'measuring cup',
    648: 'medicine chest, medicine cabinet',
    649: 'megalith, megalithic structure',
    650: 'microphone, mike',
    651: 'microwave, microwave oven',
    652: 'military uniform',
    653: 'milk can',
    654: 'minibus',
    655: 'miniskirt, mini',
    656: 'minivan',
    657: 'missile',
    658: 'mitten',
    659: 'mixing bowl',
    660: 'mobile home, manufactured home',
    661: 'Model T',
    662: 'modem',
    663: 'monastery',
    664: 'monitor',
    665: 'moped',
    666: 'mortar',
    667: 'mortarboard',
    668: 'mosque',
    669: 'mosquito net',
    670: 'motor scooter, scooter',
    671: 'mountain bike, all-terrain bike, off-roader',
    672: 'mountain tent',
    673: 'mouse, computer mouse',
    674: 'mousetrap',
    675: 'moving van',
    676: 'muzzle',
    677: 'nail',
    678: 'neck brace',
    679: 'necklace',
    680: 'nipple',
    681: 'notebook, notebook computer',
    682: 'obelisk',
    683: 'oboe, hautboy, hautbois',
    684: 'ocarina, sweet potato',
    685: 'odometer, hodometer, mileometer, milometer',
    686: 'oil filter',
    687: 'organ, pipe organ',
    688: 'oscilloscope, scope, cathode-ray oscilloscope, CRO',
    689: 'overskirt',
    690: 'oxcart',
    691: 'oxygen mask',
    692: 'packet',
    693: 'paddle, boat paddle',
    694: 'paddlewheel, paddle wheel',
    695: 'padlock',
    696: 'paintbrush',
    697: "pajama, pyjama, pj's, jammies",
    698: 'palace',
    699: 'panpipe, pandean pipe, syrinx',
    700: 'paper towel',
    701: 'parachute, chute',
    702: 'parallel bars, bars',
    703: 'park bench',
    704: 'parking meter',
    705: 'passenger car, coach, carriage',
    706: 'patio, terrace',
    707: 'pay-phone, pay-station',
    708: 'pedestal, plinth, footstall',
    709: 'pencil box, pencil case',
    710: 'pencil sharpener',
    711: 'perfume, essence',
    712: 'Petri dish',
    713: 'photocopier',
    714: 'pick, plectrum, plectron',
    715: 'pickelhaube',
    716: 'picket fence, paling',
    717: 'pickup, pickup truck',
    718: 'pier',
    719: 'piggy bank, penny bank',
    720: 'pill bottle',
    721: 'pillow',
    722: 'ping-pong ball',
    723: 'pinwheel',
    724: 'pirate, pirate ship',
    725: 'pitcher, ewer',
    726: "plane, carpenter's plane, woodworking plane",
    727: 'planetarium',
    728: 'plastic bag',
    729: 'plate rack',
    730: 'plow, plough',
    731: "plunger, plumber's helper",
    732: 'Polaroid camera, Polaroid Land camera',
    733: 'pole',
    734: 'police van, police wagon, paddy wagon, patrol wagon, wagon, black Maria',
    735: 'poncho',
    736: 'pool table, billiard table, snooker table',
    737: 'pop bottle, soda bottle',
    738: 'pot, flowerpot',
    739: "potter's wheel",
    740: 'power drill',
    741: 'prayer rug, prayer mat',
    742: 'printer',
    743: 'prison, prison house',
    744: 'projectile, missile',
    745: 'projector',
    746: 'puck, hockey puck',
    747: 'punching bag, punch bag, punching ball, punchball',
    748: 'purse',
    749: 'quill, quill pen',
    750: 'quilt, comforter, comfort, puff',
    751: 'racer, race car, racing car',
    752: 'racket, racquet',
    753: 'radiator',
    754: 'radio, wireless',
    755: 'radio telescope, radio reflector',
    756: 'rain barrel',
    757: 'recreational vehicle, RV, R.V.',
    758: 'reel',
    759: 'reflex camera',
    760: 'refrigerator, icebox',
    761: 'remote control, remote',
    762: 'restaurant, eating house, eating place, eatery',
    763: 'revolver, six-gun, six-shooter',
    764: 'rifle',
    765: 'rocking chair, rocker',
    766: 'rotisserie',
    767: 'rubber eraser, rubber, pencil eraser',
    768: 'rugby ball',
    769: 'rule, ruler',
    770: 'running shoe',
    771: 'safe',
    772: 'safety pin',
    773: 'saltshaker, salt shaker',
    774: 'sandal',
    775: 'sarong',
    776: 'sax, saxophone',
    777: 'scabbard',
    778: 'scale, weighing machine',
    779: 'school bus',
    780: 'schooner',
    781: 'scoreboard',
    782: 'screen, CRT screen',
    783: 'screw',
    784: 'screwdriver',
    785: 'seat belt, seatbelt',
    786: 'sewing machine',
    787: 'shield, buckler',
    788: 'shoe shop, shoe-shop, shoe store',
    789: 'shoji',
    790: 'shopping basket',
    791: 'shopping cart',
    792: 'shovel',
    793: 'shower cap',
    794: 'shower curtain',
    795: 'ski',
    796: 'ski mask',
    797: 'sleeping bag',
    798: 'slide rule, slipstick',
    799: 'sliding door',
    800: 'slot, one-armed bandit',
    801: 'snorkel',
    802: 'snowmobile',
    803: 'snowplow, snowplough',
    804: 'soap dispenser',
    805: 'soccer ball',
    806: 'sock',
    807: 'solar dish, solar collector, solar furnace',
    808: 'sombrero',
    809: 'soup bowl',
    810: 'space bar',
    811: 'space heater',
    812: 'space shuttle',
    813: 'spatula',
    814: 'speedboat',
    815: "spider web, spider's web",
    816: 'spindle',
    817: 'sports car, sport car',
    818: 'spotlight, spot',
    819: 'stage',
    820: 'steam locomotive',
    821: 'steel arch bridge',
    822: 'steel drum',
    823: 'stethoscope',
    824: 'stole',
    825: 'stone wall',
    826: 'stopwatch, stop watch',
    827: 'stove',
    828: 'strainer',
    829: 'streetcar, tram, tramcar, trolley, trolley car',
    830: 'stretcher',
    831: 'studio couch, day bed',
    832: 'stupa, tope',
    833: 'submarine, pigboat, sub, U-boat',
    834: 'suit, suit of clothes',
    835: 'sundial',
    836: 'sunglass',
    837: 'sunglasses, dark glasses, shades',
    838: 'sunscreen, sunblock, sun blocker',
    839: 'suspension bridge',
    840: 'swab, swob, mop',
    841: 'sweatshirt',
    842: 'swimming trunks, bathing trunks',
    843: 'swing',
    844: 'switch, electric switch, electrical switch',
    845: 'syringe',
    846: 'table lamp',
    847: 'tank, army tank, armored combat vehicle, armoured combat vehicle',
    848: 'tape player',
    849: 'teapot',
    850: 'teddy, teddy bear',
    851: 'television, television system',
    852: 'tennis ball',
    853: 'thatch, thatched roof',
    854: 'theater curtain, theatre curtain',
    855: 'thimble',
    856: 'thresher, thrasher, threshing machine',
    857: 'throne',
    858: 'tile roof',
    859: 'toaster',
    860: 'tobacco shop, tobacconist shop, tobacconist',
    861: 'toilet seat',
    862: 'torch',
    863: 'totem pole',
    864: 'tow truck, tow car, wrecker',
    865: 'toyshop',
    866: 'tractor',
    867: 'trailer truck, tractor trailer, trucking rig, rig, articulated lorry, semi',
    868: 'tray',
    869: 'trench coat',
    870: 'tricycle, trike, velocipede',
    871: 'trimaran',
    872: 'tripod',
    873: 'triumphal arch',
    874: 'trolleybus, trolley coach, trackless trolley',
    875: 'trombone',
    876: 'tub, vat',
    877: 'turnstile',
    878: 'typewriter keyboard',
    879: 'umbrella',
    880: 'unicycle, monocycle',
    881: 'upright, upright piano',
    882: 'vacuum, vacuum cleaner',
    883: 'vase',
    884: 'vault',
    885: 'velvet',
    886: 'vending machine',
    887: 'vestment',
    888: 'viaduct',
    889: 'violin, fiddle',
    890: 'volleyball',
    891: 'waffle iron',
    892: 'wall clock',
    893: 'wallet, billfold, notecase, pocketbook',
    894: 'wardrobe, closet, press',
    895: 'warplane, military plane',
    896: 'washbasin, handbasin, washbowl, lavabo, wash-hand basin',
    897: 'washer, automatic washer, washing machine',
    898: 'water bottle',
    899: 'water jug',
    900: 'water tower',
    901: 'whiskey jug',
    902: 'whistle',
    903: 'wig',
    904: 'window screen',
    905: 'window shade',
    906: 'Windsor tie',
    907: 'wine bottle',
    908: 'wing',
    909: 'wok',
    910: 'wooden spoon',
    911: 'wool, woolen, woollen',
    912: 'worm fence, snake fence, snake-rail fence, Virginia fence',
    913: 'wreck',
    914: 'yawl',
    915: 'yurt',
    916: 'web site, website, internet site, site',
    917: 'comic book',
    918: 'crossword puzzle, crossword',
    919: 'street sign',
    920: 'traffic light, traffic signal, stoplight',
    921: 'book jacket, dust cover, dust jacket, dust wrapper',
    922: 'menu',
    923: 'plate',
    924: 'guacamole',
    925: 'consomme',
    926: 'hot pot, hotpot',
    927: 'trifle',
    928: 'ice cream, icecream',
    929: 'ice lolly, lolly, lollipop, popsicle',
    930: 'French loaf',
    931: 'bagel, beigel',
    932: 'pretzel',
    933: 'cheeseburger',
    934: 'hotdog, hot dog, red hot',
    935: 'mashed potato',
    936: 'head cabbage',
    937: 'broccoli',
    938: 'cauliflower',
    939: 'zucchini, courgette',
    940: 'spaghetti squash',
    941: 'acorn squash',
    942: 'butternut squash',
    943: 'cucumber, cuke',
    944: 'artichoke, globe artichoke',
    945: 'bell pepper',
    946: 'cardoon',
    947: 'mushroom',
    948: 'Granny Smith',
    949: 'strawberry',
    950: 'orange',
    951: 'lemon',
    952: 'fig',
    953: 'pineapple, ananas',
    954: 'banana',
    955: 'jackfruit, jak, jack',
    956: 'custard apple',
    957: 'pomegranate',
    958: 'hay',
    959: 'carbonara',
    960: 'chocolate sauce, chocolate syrup',
    961: 'dough',
    962: 'meat loaf, meatloaf',
    963: 'pizza, pizza pie',
    964: 'potpie',
    965: 'burrito',
    966: 'red wine',
    967: 'espresso',
    968: 'cup',
    969: 'eggnog',
    970: 'alp',
    971: 'bubble',
    972: 'cliff, drop, drop-off',
    973: 'coral reef',
    974: 'geyser',
    975: 'lakeside, lakeshore',
    976: 'promontory, headland, head, foreland',
    977: 'sandbar, sand bar',
    978: 'seashore, coast, seacoast, sea-coast',
    979: 'valley, vale',
    980: 'volcano',
    981: 'ballplayer, baseball player',
    982: 'groom, bridegroom',
    983: 'scuba diver',
    984: 'rapeseed',
    985: 'daisy',
    986: "yellow lady's slipper, yellow lady-slipper, Cypripedium calceolus, Cypripedium parviflorum",
    987: 'corn',
    988: 'acorn',
    989: 'hip, rose hip, rosehip',
    990: 'buckeye, horse chestnut, conker',
    991: 'coral fungus',
    992: 'agaric',
    993: 'gyromitra',
    994: 'stinkhorn, carrion fungus',
    995: 'earthstar',
    996: 'hen-of-the-woods, hen of the woods, Polyporus frondosus, Grifola frondosa',
    997: 'bolete',
    998: 'ear, spike, capitulum',
    999: 'toilet tissue, toilet paper, bathroom tissue'}
    

class ContrastiveTransformations:
    def __init__(self, base_transforms, n_views=2, spe_transforms=None):
        self.base_transforms = base_transforms
        self.n_views = n_views
        self.spe_transforms = spe_transforms

    def __call__(self, x):
        if self.spe_transforms is None: return [self.base_transforms(x) for i in range(self.n_views)] # return n_views versions of an image following base_transforms
        else: return [self.base_transforms(x), self.spe_transforms(x)] # return only pair of [base_transforms, specific_transforms] augmented image

def simclr_aug(size, *args, norm):
    '''
    args of form:  (aug1, p1), (aug2, p2), (aug3, p3), ... 
    Note - RandomResizedCrop & GaussianBlur do not have p as parameter, must pass:  (aug, None)
    Possible examples:
        (hflip, p)
        (crop, None)
        (c_jitter, p)
        (gray, p)
        (g_blur, None)
    '''
    # Return standard simCLR augmentations
    if not args:

        return [transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(size=size),
                transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.GaussianBlur(kernel_size=9),
                transforms.ToTensor(),
                transforms.Normalize(norm[0], norm[1]),]
   
    # Create custom list of augmentations and probabilities
    else:
        aug = []
        
        if size == 224: # If ImageNet, some images are not of same size so must apply resize for uniformity
            aug += [transforms.Resize(256), 
                    transforms.CenterCrop(224)]

        for des_aug in args:
            if des_aug[0] == 'hflip':
                aug += [transforms.RandomHorizontalFlip(p=des_aug[1])]
            
            elif des_aug[0] == 'crop':
                aug += [transforms.RandomResizedCrop(size=size)]
            
            elif des_aug[0] == 'c_jitter':
                aug += [transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)], p=des_aug[1])]

            elif des_aug[0] == 'gray':
                aug += [transforms.RandomGrayscale(p=des_aug[1])]

            elif des_aug[0] == 'g_blur':
                aug += [transforms.GaussianBlur(kernel_size=9)]
        
        aug += [transforms.ToTensor(),
                transforms.Normalize(norm[0], norm[1]),]

        return aug

def load_data(dataset, *args_simclr, bs=64, stage='pre', finetune=False, n_views=1, 
              spe_pair=False, aug=False,                                                                      # simclr
              pre_type='supervised', noise_path=None, fractal_path=None, resize_image=False, shuffle_noise=True, random_seed=42, # counterfact & noise & fractal
              jigsaw_ps=None, # mix patches
              num_workers=4):
    # TODO: pin_memory = True
    cifar_norm = [(0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)]
    ImageNet_norm = [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]
    transform_array = []

    if dataset=='CIFAR10':
        transform_array += [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar_norm[0],
                                  std=cifar_norm[1])]
        transform = transforms.Compose(transform_array)
        
        if aug: # Augment the dataset with simCLR aug
            transform = transforms.Compose(simclr_aug(size=32, norm=cifar_norm))
        
        elif jigsaw_ps is not None: #TODO: Normalize each patch independently?
            shuffle = ShufflePatches(jigsaw_ps)
            
            transform_array = [transforms.ToTensor(),
                               shuffle,
                               transforms.Normalize(mean=cifar_norm[0],
                                                    std=cifar_norm[1])]
            transform = transforms.Compose(transform_array)

        elif n_views > 1: # Apply SimCLR augmentations
            if not args_simclr: # standard aug for any nb of views
                transform_array = simclr_aug(size=32, norm=cifar_norm)
                transform = ContrastiveTransformations(transforms.Compose(transform_array), n_views=n_views)
            elif spe_pair: # custom aug for 2 views of (non-aug, aug) combinaiton
                spe_transform_array = simclr_aug(32, args_simclr, norm=cifar_norm)
                transform = ContrastiveTransformations(transforms.Compose(transform_array),
                                                       spe_transforms=transforms.Compose(spe_transform_array))
            else: # custom aug for any nb of views
                transform_array = simclr_aug(32, args_simclr, norm=cifar_norm)
                transform = ContrastiveTransformations(transforms.Compose(transform_array), n_views=n_views)

        
        # TODO: WILL downstream without finetuning ever happen??
        # On downstream without finetuning, no need to load train and test sets separately, 
        # for bigger test set, set train=True
        if stage=='down' and not finetune:
            ds_test = CIFAR10(root='./data', train=False, download=True, transform=transform)
            test_loader = DataLoader(ds_test, batch_size=bs, shuffle=False, num_workers=num_workers)

            return None, test_loader

        else:
            ds_train = CIFAR10(root='./data', train=True, download=True, transform=transform)
            ds_test = CIFAR10(root='./data', train=False, download=True, transform=transform)
            train_loader = DataLoader(ds_train, batch_size=bs, shuffle=True, num_workers=num_workers)
            test_loader = DataLoader(ds_test, batch_size=bs, shuffle=False, num_workers=num_workers)

            return train_loader, test_loader
    
    elif dataset=='ImageNet': # TODO: load test manually
        transform_array += [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=ImageNet_norm[0],
                                 std=ImageNet_norm[1])]
        transform = transforms.Compose(transform_array)

        if aug: # Augment the dataset with simCLR aug
            transform = transforms.Compose(simclr_aug(size=224, norm=ImageNet_norm))
        
        elif jigsaw_ps is not None:
            shuffle = ShufflePatches(jigsaw_ps)
            
            transform_array = [transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               shuffle,
                               transforms.Normalize(mean=ImageNet_norm[0],
                                                    std=ImageNet_norm[1])]
            transform = transforms.Compose(transform_array)

        elif n_views > 1: # Apply SimCLR augmentations
            if not args_simclr:
                transform_array = simclr_aug(size=224, norm=ImageNet_norm)
                transform = ContrastiveTransformations(transforms.Compose(transform_array), n_views=n_views)
            elif spe_pair: # test embed distance for specific pair of normal/augmented combinaiton
                spe_transform_array = simclr_aug(224, args_simclr, norm=ImageNet_norm)
                transform = ContrastiveTransformations(transforms.Compose(transform_array), n_views=n_views, 
                                                       spe_transforms=transforms.Compose(spe_transform_array))
            else: # custom set of augmentations
                transform_array = simclr_aug(224, args_simclr, norm=ImageNet_norm)
                transform = ContrastiveTransformations(transforms.Compose(transform_array), n_views=n_views)
        
        # On downstream without finetuning, no need to load train and test sets separately, for more exhaustive test set, set train=True
        if stage=='down' and not finetune:
            ds_test = ImageNet(root='./data', split='val', transform=transform)
            test_loader = DataLoader(ds_test, batch_size=bs, shuffle=False, num_workers=num_workers)

            return None, test_loader
        
        else:
            ds_train = ImageNet(root='./data', split='train', transform=transform)
            ds_test = ImageNet(root='./data', split='val', transform=transform)
            train_loader = DataLoader(ds_train, batch_size=bs, shuffle=True, num_workers=num_workers)
            test_loader = DataLoader(ds_test, batch_size=bs, shuffle=False, num_workers=num_workers)

            return train_loader, test_loader

    elif dataset=='tiny': #TODO: what are standard augs on tiny?
        transform_array += [
            transforms.ToTensor(),            
            transforms.Normalize(mean=ImageNet_norm[0],
                                 std=ImageNet_norm[1])]
        transform = transforms.Compose(transform_array)

        if aug: # Augment the dataset with simCLR aug
            transform = transforms.Compose(simclr_aug(size=64, norm=ImageNet_norm))
        
        elif jigsaw_ps is not None:
            shuffle = ShufflePatches(jigsaw_ps)
            
            transform_array = [transforms.ToTensor(),
                               shuffle,
                               transforms.Normalize(mean=ImageNet_norm[0],
                                                    std=ImageNet_norm[1])]
            transform = transforms.Compose(transform_array)

        elif n_views > 1: # Apply SimCLR augmentations
            if not args_simclr:
                transform_array = simclr_aug(size=64, norm=ImageNet_norm)
                transform = ContrastiveTransformations(transforms.Compose(transform_array), n_views=n_views)
            elif spe_pair: # test embed distance for specific pair of normal/augmented combinaiton
                spe_transform_array = simclr_aug(64, args_simclr, norm=ImageNet_norm)
                transform = ContrastiveTransformations(transforms.Compose(transform_array), n_views=n_views, 
                                                       spe_transforms=transforms.Compose(spe_transform_array))
            else: # custom set of augmentations
                transform_array = simclr_aug(64, args_simclr, norm=ImageNet_norm)
                transform = ContrastiveTransformations(transforms.Compose(transform_array), n_views=n_views)

        # On downstream without finetuning, no need to load train and test sets separately, for more exhaustive test set, set train=True
        if stage=='down' and not finetune:
            ds_test = TinyImageNetDataset(stage='val', transform=transform)
            test_loader = DataLoader(ds_test, batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=True)

            return None, test_loader
        
        else:
            ds_train = TinyImageNetDataset(stage='train', transform=transform)
            ds_test = TinyImageNetDataset(stage='val', transform=transform)
            train_loader = DataLoader(ds_train, batch_size=bs, shuffle=True, num_workers=num_workers, pin_memory=True)
            test_loader = DataLoader(ds_test, batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=True)

            return train_loader, test_loader
    
    elif dataset=='STL10':

        contrast_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=96),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=9),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),])

        # On downstream without finetuning, no need to load train and test sets separately, 
        # for bigger test set, set split=unlabeled
        if stage=='down' and not finetune:
            labeled_data_contrast = STL10(
                root='./data',
                split="train",
                download=True,
                transform=ContrastiveTransformations(contrast_transforms, n_views=2))
            test_loader = DataLoader(labeled_data_contrast, batch_size=bs, shuffle=False, num_workers=num_workers)

            return None, test_loader

        else:
            unlabeled_data = STL10(
                root='./data',
                split="unlabeled",
                download=True,
                transform=ContrastiveTransformations(contrast_transforms, n_views=2))
            labeled_data_contrast = STL10(
                root='./data',
                split="train",
                download=True,
                transform=ContrastiveTransformations(contrast_transforms, n_views=2))
            train_loader = DataLoader(unlabeled_data, batch_size=bs, shuffle=True, num_workers=num_workers)
            test_loader = DataLoader(labeled_data_contrast, batch_size=bs, shuffle=False, num_workers=num_workers)

            return train_loader, test_loader
    
    elif dataset=='cityscapes':
        # TODO: check if transforms is right of if more needed, SEE downstream.down_seg
        # TODO: MIGHT BE OUTDATED WITH REST OF CODE
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
        ])

        ds_train = Cityscapes(root='./data/cityscapes', split='train', mode='fine', target_type='semantic', download=True, transform=transform)
        ds_val = Cityscapes(root='./data/cityscapes', split='val', mode='fine', target_type='semantic', download=True, transform=transform)
        ds_test = Cityscapes(root='./data/cityscapes', split='test', mode='fine', target_type='semantic', download=True, transform=transform)
        train_loader = DataLoader(ds_train, batch_size=bs, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(ds_val, batch_size=bs, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(ds_test, batch_size=bs, shuffle=False, num_workers=num_workers)

        return train_loader, val_loader, test_loader


    elif dataset=='OTHER CUSTOM ONE':
        pass

    elif dataset=='counterfactual':
        if resize_image:
            transform_array.append(
                transforms.Resize((96,96))
            )

        transform_array += [
            transforms.RandomResizedCrop(64, scale=(0.08, 1)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=ImageNet_norm[0],
                                 std=ImageNet_norm[1]),]

        transform = ContrastiveTransformations(transforms.Compose(transform_array), n_views=n_views)

        df, size = load_counterfact()
        dataset = Dataset_counterfact(df=df, transforms=transform, size=size, pre_type=pre_type)

        # Split training and validation splits:
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        test_size = 0.1
        split = int(np.floor(test_size * dataset_size))
        if shuffle_noise :
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        train_loader = DataLoader(dataset, batch_size=bs, num_workers=num_workers,
                                  shuffle=False, pin_memory=True, sampler=SubsetRandomSampler(train_indices))
        test_loader = DataLoader(dataset, batch_size=bs, num_workers=num_workers,
                                 pin_memory=True, sampler=SubsetRandomSampler(val_indices))
        
        return train_loader, test_loader

    elif dataset=='noise':
        if resize_image:
            transform_array.append(
                transforms.Resize((96,96))
            )

        transform_array += [
            transforms.RandomResizedCrop(64, scale=(0.08, 1)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.44087801806139126, 0.42790631331699347, 0.3867879370752931),
                (0.26826768628079806, 0.2610450402318512, 0.26866836876860795),
            ),]

        transform = ContrastiveTransformations(transforms.Compose(transform_array), n_views=n_views)
        # dataset = datasets.ImageFolder('./data/noise', transform=transform)
        dataset = Dataset_no_label(noise_path, transforms=transform)

        # Split training and validation splits:
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        test_size = 0.1
        split = int(np.floor(test_size * dataset_size))
        if shuffle_noise :
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]


        train_loader = DataLoader(dataset, batch_size=bs, num_workers=num_workers,
                                                   shuffle=False, pin_memory=True, sampler=SubsetRandomSampler(train_indices))
        test_loader = DataLoader(dataset, batch_size=bs, num_workers=num_workers,
                                                  pin_memory=True, sampler=SubsetRandomSampler(val_indices))
        
        return train_loader, test_loader
    
    elif dataset=='fractal':
        if resize_image:
            transform_array.append(
                transforms.Resize((96,96))
            )

        transform_array += [
            transforms.RandomResizedCrop(64, scale=(0.08, 1)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),] #TODO: Normalize?

        transform = ContrastiveTransformations(transforms.Compose(transform_array), n_views=n_views)
        dataset = Dataset_no_label(fractal_path, transforms=transform)

        # Split training and validation splits:
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        test_size = 0.1
        split = int(np.floor(test_size * dataset_size))
        if shuffle_noise :
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]


        train_loader = DataLoader(dataset, batch_size=bs, num_workers=num_workers,
                                                   shuffle=False, pin_memory=True, sampler=SubsetRandomSampler(train_indices))
        test_loader = DataLoader(dataset, batch_size=bs, num_workers=num_workers,
                                                  pin_memory=True, sampler=SubsetRandomSampler(val_indices))
        
        return train_loader, test_loader

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, img_list, transforms):
        super(MyDataset, self).__init__()
        self.img_list = img_list
        self.transforms = transforms

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        title = self.img_list[idx]
        img = Image.open(title)
        return self.transforms(img), title

class Dataset_counterfact(torch.utils.data.Dataset):
    def __init__(self, df, transforms, size, pre_type):
        self.df = df
        self.transforms = transforms
        self.size = size
        self.pre_type = pre_type

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        title = os.path.join("./data", "{}_imgs".format(self.size), "ims", self.df.im_name[index]) + '_x_gen.jpg'
        image = Image.open(title)
        
        if self.pre_type == 'contrastive': # do not return labels
            return self.transforms(image)

        elif self.pre_type == 'shape_bias': #TODO: will only need these if want to use to eval shape bias
            labels = (self.df.shape_cls[index], self.df.texture_cls[index], self.df.bg_cls[index])
        else:
            labels = self.df.shape_cls[index]

        return self.transforms(image), labels

class Dataset_no_label(torch.utils.data.Dataset):
    def __init__(self, img_list, transforms):
        self.img_list = img_list
        self.transforms = transforms

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx])
        return self.transforms(img) # no labels for this dataset


def tiny_class_to_int():
    classes = open("data/tiny-imagenet-200/wnids.txt").readlines()
    map = {}
    i = 0
    for line in classes:
        map[line.strip('\n')] = i
        i += 1
    return map

class TinyImageNetDataset(torch.utils.data.Dataset):
    # Adapted from from https://www.kaggle.com/c/thu-deep-learning/overview/tips
    def __init__(self, root="./data/tiny-imagenet-200", stage='train', transform=transforms.ToTensor()):
        # root: your_path/TinyImageNet/
        images = []
        map = tiny_class_to_int()

        if stage=='train':
            path = "./data/tiny-imagenet-200/train"
            for label in os.listdir(path):
                for image in os.listdir(os.path.join(path, label, 'images')):
                    name = os.path.join('train', label, 'images', image)
                    images.append((name, map[label]))

        elif stage=='val':
            path = "./data/tiny-imagenet-200/val/val_annotations.txt"
            labels = open(path).readlines()
            for line in labels:
                items = line.strip('\n').split()
                img_name = os.path.join('val', 'images', items[0])

                # test list contains only image name
                test_flag = True if len(items) == 1 else False
                label = None if test_flag == True else np.array(map[items[1]])

                if os.path.isfile(os.path.join(root, img_name)):
                    images.append((img_name, label))
                else:
                    print(os.path.join(root, img_name) + 'Not Found.')

        self.root = root
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_name, label = self.images[index]
        img = Image.open(os.path.join(self.root, img_name)).convert('RGB')

        return self.transform(img), label


def load_noise_brute():
    noise_list = [
        'dead_leaves-squares',
        'dead_leaves-oriented',
        'dead_leaves-mixed',
        'dead_leaves-textures',
        'stat-spectrum',
        'stat-wmm',
        'stat-spectrum_color',
        'stat-spectrum_color_wmm',
        'stylegan-random',
        'stylegan-highfreq',
        'stylegan-sparse',
        'stylegan-oriented',
        'feature_vis-random',
        'feature_vis-dead_leaves'
    ]

    paths = []
    for folder in noise_list:
        for subfolder in np.arange(105):
            subfolder = str(subfolder).zfill(7) + '000'
            for image in range(1000):
                paths.append('./data/noise/{}/{}/{}.jpg'.format(folder, subfolder, str(image).zfill(10)))

    return paths

def load_noise(*args):
    if not args:
        noise_list = [
        'dead_leaves-squares',
        'dead_leaves-oriented',
        'dead_leaves-mixed',
        'dead_leaves-textures',
        'stat-spectrum',
        'stat-wmm',
        'stat-spectrum_color',
        'stat-spectrum_color_wmm',
        'stylegan-random',
        'stylegan-highfreq',
        'stylegan-sparse',
        'stylegan-oriented',
        'feature_vis-random',
        'feature_vis-dead_leaves']
    else:
        noise_list = []
        for i in args:
            noise_list.append(i)

    paths = []
    for folder in noise_list:
        for subfolder in os.listdir(os.path.join("data", "noise", folder)):
            if not subfolder.isdigit(): continue
            for image in os.listdir(os.path.join("data", "noise", folder, subfolder)):
                paths.append(os.path.join("data", "noise", folder, subfolder, image))

    return paths

def load_fractal():
    # TODO: modif to select subset of images
    fractal_path = os.path.join("data", "FractalDB")
    paths = []
    for folder in os.listdir(fractal_path):
        for img in os.listdir(os.path.join(fractal_path, folder)):
            paths.append(os.path.join(fractal_path, folder, img))
    
    return paths

def load_geirhos_transfer_pre(conflict_only=False):
    paths = []
    for category in os.listdir(os.path.join("geirhos", "style-transfer-preprocessed-512")):
        if category == ".DS_Store":
            continue
        
        for img in os.listdir(os.path.join("geirhos", "style-transfer-preprocessed-512", category)):
            
            # disregard images where shape and texture are identical
            if conflict_only:
                name_split  = img.split("-")
                shape = name_split[0]
                texture = name_split[1]
                shape = shape[:-2] if shape[-2].isdigit() else shape[:-1]
                texture = texture[:-6] if texture[-6].isdigit() else texture[:-5]

                if shape == texture:
                    continue
                
            paths.append(os.path.join("geirhos", "style-transfer-preprocessed-512", category, img))
    
    return paths

def load_counterfact(size=10000, verbose=False):
    avail_sizes = [10, 500, 2500, 10000]

    # Select dataset to best fit nb of images required
    if size <= avail_sizes[0]:
        to_pick = avail_sizes[0]
    elif size >= avail_sizes[-1]:
        to_pick = avail_sizes[-1]
    else:
        for i in range(0, len(avail_sizes)-1):
            if avail_sizes[i] < size <= avail_sizes[i+1]:
                to_pick = avail_sizes[i+1]
    
    dir = "./data/{}_imgs".format(to_pick)    
    df = pd.read_csv(os.path.join(dir, 'labels.csv'))
    df = df.drop(df.columns[0], axis=1)

    if verbose: print('Loading counterfactual set of size {}'.format(to_pick))
    return df, to_pick


class ShufflePatches(object):
  def __init__(self, patch_size):
    self.ps = patch_size

  def __call__(self, x):
    '''
    Adapted from https://stackoverflow.com/questions/66962837/shuffle-patches-in-image-batch
    '''
    # divide the batch of images into non-overlapping patches
    u = F.unfold(x, kernel_size=self.ps, stride=self.ps, padding=0)
    # permute the patches of each image in the batch
    u = torch.unsqueeze(u, 0)
    pu = torch.cat([b_[:,torch.randperm(b_.shape[-1])] for b_ in u], dim=0)
    # fold the permuted patches back together
    f = F.fold(pu, x.shape[-2:], kernel_size=self.ps, stride=self.ps, padding=0)
    return f 

if __name__ == '__main__':
    pass
