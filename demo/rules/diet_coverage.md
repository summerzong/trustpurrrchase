# Diet rule coverage

Generated 2026-11-14T19:24:00+02:00 from 17546 products by `build_diet_vocab.py`.

Each rule is run on every product. *Strong* = structured evidence without contradiction; *conflict* = structured evidence contradicted by another source (sent to a person); *name only* = weak evidence (sent to a person). Rules with reviewed = no can be searched, but an order that relies on them always needs a person.

| code | label | reviewed | GS1 claims | strong | conflict | name only | look-alike | buyer terms | suggested contradictions (ratio) | policy contradictions |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| `FREE_FROM_LACTOSE` | lactose-free | yes | 4178 | 4628 | 49 | 14 | 764 | 6 | ML 0.037 | ML |
| `FREE_FROM_GLUTEN` | gluten-free | no | 3049 | 5185 | 42 | 12 | 3076 | 10 | AW 0.023, NR 0.027, UW 0.068 | AW, NR, GB (+MAY_CONTAIN) |
| `VEGAN` | vegan | no | 2959 | 2952 | 7 | 8 | 4054 | 3 | AE 0, AF 0, AM 0, ML 0.092 | AM, ML, AE, AF, AC, UM |
| `VEGETARIAN` | vegetarian | no | 2358 | 4242 | 1 | 79 | 521 | 3 | AF 0.016 | AF, AC, UM |
| `FREE_FROM_PRESERVATIVES` | free from preservatives | no | 1526 | 1526 | 0 | 0 | 0 | 4 | – | – |
| `FREE_FROM_COLOURING_AGENTS` | free from colouring agents | no | 932 | 932 | 0 | 2 | 0 | 5 | – | – |
| `FREE_FROM_ADDITIVES` | free from additives | no | 690 | 690 | 0 | 1 | 0 | 5 | – | – |
| `NO_ADDED_SUGARS` | no added sugar | no | 662 | 662 | 0 | 3 | 0 | 4 | – | – |
| `HIGH_FIBRE` | high-fibre | no | 420 | 1213 | 0 | 0 | 0 | 7 | – | – |
| `FREE_FROM_EGGS` | egg-free | no | 418 | 4296 | 27 | 0 | 1864 | 11 | AE 0 | AE (+MAY_CONTAIN) |
| `HIGH_PROTEIN` | high-protein | no | 417 | 417 | 0 | 0 | 0 | 5 | – | – |
| `LOW_SODIUM_SALT` | low-salt | no | 390 | 5391 | 28 | 2 | 6533 | 5 | – | – |
| `FREE_FROM_SWEETENERS` | free from sweeteners | no | 358 | 358 | 0 | 4 | 0 | 4 | ML 0, AY 0.089 | – |
| `FREE_FROM_SOY` | soy-free | no | 337 | 4184 | 7 | 0 | 2051 | 10 | AY 0.047 | AY (+MAY_CONTAIN) |
| `LOW_LACTOSE` | low-lactose | yes | 321 | 5005 | 3 | 19 | 158 | 3 | – | – |
| `FREE_FROM_FAT` | fat-free | no | 158 | 3027 | 10 | 0 | 9116 | 0 | only 158 products: sample too small | – |
| `FREE_FROM_WHEAT` | wheat-free | no | 141 | 643 | 0 | 1 | 2898 | 9 | only 141 products: sample too small | UW (+MAY_CONTAIN) |
| `LOW_FAT` | low-fat | no | 110 | 5056 | 10 | 0 | 7087 | 0 | – | – |
| `FREE_FROM_ARTIFICIAL_SWEETENERS` | free from artificial sweeteners | no | 24 | 24 | 0 | 0 | 0 | 2 | only 24 products: sample too small | – |
| `FREE_FROM_GMO` | free from gmo | no | 14 | 14 | 0 | 0 | 0 | 4 | only 14 products: sample too small | – |
| `FREE_FROM_SODIUM_SALT` | salt-free | no | 13 | 1858 | 67 | 2 | 10027 | 6 | only 13 products: sample too small | – |
| `CELERY_FREE` | celery-free | no | 0 | 4560 | 0 | 2 | 301 | 9 | – | BC (+MAY_CONTAIN) |
| `CRUSTACEAN_FREE` | crustacean-free | no | 0 | 4632 | 0 | 0 | 94 | 12 | – | AC (+MAY_CONTAIN) |
| `DAIRY_FREE` | dairy-free | no | 0 | 3583 | 3 | 6 | 3898 | 11 | – | AM, ML (+MAY_CONTAIN) |
| `FISH_FREE` | fish-free | no | 0 | 4518 | 0 | 0 | 525 | 9 | – | AF (+MAY_CONTAIN) |
| `LUPIN_FREE` | lupin-free | no | 0 | 4629 | 0 | 0 | 121 | 8 | – | NL (+MAY_CONTAIN) |
| `MOLLUSC_FREE` | mollusc-free | no | 0 | 4651 | 0 | 0 | 66 | 12 | – | UM (+MAY_CONTAIN) |
| `MUSTARD_FREE` | mustard-free | no | 0 | 4270 | 0 | 0 | 867 | 9 | – | BM (+MAY_CONTAIN) |
| `NUT_FREE` | nut-free | no | 0 | 4196 | 1 | 1 | 1534 | 11 | – | AN, AP (+MAY_CONTAIN) |
| `PEANUT_FREE` | peanut-free | no | 0 | 4501 | 0 | 0 | 510 | 10 | – | AP (+MAY_CONTAIN) |
| `SESAME_FREE` | sesame-free | no | 0 | 4353 | 0 | 0 | 756 | 9 | – | AS (+MAY_CONTAIN) |
| `SULPHITE_FREE` | sulphite-free | no | 0 | 4563 | 0 | 0 | 332 | 12 | – | AU (+MAY_CONTAIN) |

Ratio = how often the allergen is listed as CONTAINS under the claim, relative to the whole catalogue (0 = never). A low ratio suggests the allergen contradicts the claim; a person decides whether to copy it into `diet_overrides.json`.
