# GT Audit — Cross-validation hallucinations

Models analyzed: ['p5j_r32_lr7e5_ep5_a256', 'p4h_r32_lr2e4_ep5', 'p4g_r48_lr1e4_ep5', 'p4g_r16_lr1e4_ep8']

Images compared: 392


## HALLUC_CONSENSUS (51 cases)

gt = null, но ≥2 модели независимо вывели одинаковое значение → возможно gt неполное.


### По полям:

- **type_status**: 13
- **verbatim_collector**: 6
- **collector**: 6
- **exsiccata_title**: 4
- **verbatim_latitude**: 3
- **verbatim_longitude**: 3
- **verbatim_elevation**: 3
- **exsiccata_number**: 3
- **field_number**: 2
- **determiner**: 2
- **verbatim_collection_date**: 2
- **organism_remarks**: 2
- **verbatim_substrate**: 1
- **verbatim_habitat**: 1

### Топ-30 кандидатов на исправление gt (сортировка по n_agreeing):

| Image | Field | n_models | Consensus value |
|---|---|---|---|
| LE F-66924.JPG | collector | 4 | 'Kühn J. G.' |
| LE F-61859.JPG | exsiccata_number | 4 | '375' |
| LE F-227006.JPG | field_number | 4 | '5586' |
| LE F-262338.JPG | field_number | 4 | '5794' |
| LE F-63688.JPG | organism_remarks | 4 | 'etiam in parte sepulta, haud chlorophylliphera' |
| LE F-63689.JPG | organism_remarks | 4 | 'etiam in parte sepulta, haud chlorophylliphera' |
| LE F-62024_1.JPG | type_status | 4 | 'Type' |
| LE F-62024_2.JPG | type_status | 4 | 'Type' |
| LE F-62028_2.JPG | type_status | 4 | 'Type' |
| LE F-63641.JPG | type_status | 4 | 'Type' |
| LE F-63689.JPG | type_status | 4 | 'Type' |
| LE F-43579.JPG | verbatim_collection_date | 4 | '182' |
| LE F-43620.JPG | verbatim_collection_date | 4 | '27/1. 80' |
| LE F-343151.JPG | verbatim_collector | 4 | 'Б.К. Калымбетов' |
| LE F-43616.JPG | verbatim_collector | 4 | 'Julius Kühn' |
| LE F-66924.JPG | verbatim_collector | 4 | 'Prof. Dr. Julius Kühn' |
| LE F-214102.JPG | verbatim_elevation | 4 | 'Alt. c. 520 m' |
| LE F-68604.JPG | verbatim_elevation | 4 | 'Alt. c. 1300 m.' |
| LE F-61722.JPG | verbatim_habitat | 4 | 'in pratis' |
| LE F-214102.JPG | verbatim_latitude | 4 | "16°25' S" |
| LE F-262347.JPG | verbatim_latitude | 4 | "9°10' N" |
| LE F-68604.JPG | verbatim_latitude | 4 | "21°38' N" |
| LE F-214102.JPG | verbatim_longitude | 4 | "144°45' E" |
| LE F-262347.JPG | verbatim_longitude | 4 | "83°23' W" |
| LE F-68604.JPG | verbatim_longitude | 4 | "101°18' E" |
| LE F-43616.JPG | collector | 3 | 'Kühn J. G.' |
| LE F-62935.JPG | collector | 3 | 'Kühn J. G.' |
| LE F-68068.JPG | collector | 3 | 'Golovin P. N.' |
| LE F-43556.JPG | type_status | 3 | 'Type' |
| LE F-43557.JPG | type_status | 3 | 'Type' |

## MODEL_BIAS (115 cases)

gt ≠ null, но ≥2 модели согласны на ДРУГОМ значении.


### Топ-30 (потенциальные ошибки модели или gt):

| Image | Field | sim | n_models | GT | Consensus |
|---|---|---|---|---|---|
| LE F-43593.JPG | verbatim_collection_date | 0.00 | 4 | '[NULL]' | '182' |
| LE F-61085.JPG | type_status | 0.19 | 4 | 'Original material' | 'Type' |
| LE F-60833_2.JPG | exsiccata_number | 0.29 | 4 | '1698' | '283' |
| LE F-64889_2.JPG | exsiccata_number | 0.29 | 4 | '1698' | '283' |
| LE F-43612.JPG | country | 0.31 | 4 | 'Poland' | 'Germany' |
| LE F-61708.JPG | type_status | 0.31 | 4 | 'Authentic' | 'Type' |
| LE F-314150.JPG | verbatim_longitude | 0.43 | 4 | '141°59\'04" E' | '26°48\'12" E' |
| LE F-66924.JPG | verbatim_collection_date | 0.44 | 4 | '23. Febr. 1879' | '14. September 1878' |
| LE F-214060.JPG | type_status | 0.44 | 4 | 'Isoneotype' | 'Paratype' |
| LE F-314150.JPG | verbatim_latitude | 0.46 | 4 | '43°34\'49" N' | '16°56\'40" S' |
| LE F-65008.JPG | verbatim_locality | 0.48 | 4 | 'auf der zur Herrschaft Schwusen (Niederschlesien) ' | 'auf dem Versuchsfelde des landwirthschaftlichen In' |
| LE F-61707.JPG | exsiccata_number | 0.50 | 4 | '42' | '49' |
| LE F-43612.JPG | verbatim_substrate | 0.54 | 4 | 'an der cultivirten Form von Panicum sanguinale beo' | 'Panicum sanguinalem' |
| LE F-43493.JPG | verbatim_collection_date | 0.61 | 4 | '9.1895' | '9. 1895' |
| LE F-43577.JPG | verbatim_collection_date | 0.61 | 4 | '27. Sept. 1877' | '28. Mai 1877' |
| LE F-65008.JPG | verbatim_substrate | 0.65 | 4 | 'an Sorghum racharatum' | 'an Sorghum vulgare cultivirt' |
| LE F-227018.JPG | verbatim_longitude | 0.67 | 4 | '144°03\'21" E' | '144°51\'58" E' |
| LE F-62040.JPG | verbatim_collection_date | 0.67 | 4 | '19.VIII.1957 et 13.VIII.1959' | '19. VIII. 1957 et 13. VIII. 1959' |
| LE F-62244_1.JPG | type_status | 0.67 | 4 | 'Holotype' | 'Type' |
| LE F-62244_2.JPG | type_status | 0.67 | 4 | 'Holotype' | 'Type' |
| LE F-198015.JPG | type_status | 0.73 | 4 | 'Type' | 'Isotype' |
| LE F-22416.JPG | type_status | 0.73 | 4 | 'Isotype' | 'Type' |
| LE F-22418.JPG | type_status | 0.73 | 4 | 'Isotype' | 'Type' |
| LE F-43580.JPG | verbatim_collector | 0.73 | 4 | 'Tycho Vestergren' | 'Tycho Vester gren' |
| LE F-43622.JPG | scientific_name | 0.74 | 4 | 'Ustilago tragica Vánky' | 'Sphacelotheca tragica Vánky' |
| LE F-214149.JPG | field_number | 0.75 | 4 | '5164' | '5165' |
| LE F-67371_1.JPG | exsiccata_number | 0.75 | 4 | '1997' | '1992' |
| LE F-61500.JPG | verbatim_identification | 0.77 | 4 | 'Entyloma primulae Murashk. nov. sp.' | 'Entyloma primulae Woronich.' |
| LE F-214028.JPG | verbatim_latitude | 0.78 | 4 | "25°47'06'' N" | '25°47\'06" S' |
| LE F-214028.JPG | verbatim_longitude | 0.78 | 4 | "28°10'38'' W" | '28°10\'38" E' |
