# Per-field comparison — top models on 394 full corpus

Models: p4h_r32_lr2e4_ep5, p5j_r32_lr7e5_ep5_a256, p4g_r48_lr1e4_ep5, p4g_r16_lr1e4_ep8

## F1 per field — все модели

| Field | n_gt | F1 p4h_r32_lr2e4_ep5 | F1 p5j_r32_lr7e5_ep5_a256 | F1 p4g_r48_lr1e4_ep5 | F1 p4g_r16_lr1e4_ep8 |
|---|---|---|---|---|---|
| verbatim_identification_date | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| identification_remarks | 4 | 0.400 | 0.500 | 0.000 | 0.400 |
| organism_remarks | 11 | 0.762 | 0.727 | 0.556 | 0.870 |
| verbatim_habitat | 32 | 0.852 | 0.867 | 0.814 | 0.814 |
| collector | 370 | 0.960 | 0.955 | 0.961 | 0.952 |
| field_number | 124 | 0.950 | 0.954 | 0.964 | 0.960 |
| verbatim_locality | 377 | 0.960 | 0.961 | 0.957 | 0.953 |
| verbatim_collection_date | 374 | 0.963 | 0.957 | 0.961 | 0.958 |
| verbatim_collector | 370 | 0.967 | 0.962 | 0.962 | 0.949 |
| determiner | 371 | 0.968 | 0.965 | 0.965 | 0.963 |
| type_status | 362 | 0.964 | 0.969 | 0.962 | 0.967 |
| verbatim_longitude | 131 | 0.969 | 0.973 | 0.970 | 0.966 |
| verbatim_substrate | 374 | 0.974 | 0.971 | 0.971 | 0.969 |
| scientific_name | 378 | 0.970 | 0.974 | 0.975 | 0.966 |
| verbatim_latitude | 131 | 0.973 | 0.977 | 0.970 | 0.970 |
| verbatim_identification | 378 | 0.977 | 0.975 | 0.977 | 0.969 |
| exsiccata_number | 322 | 0.980 | 0.984 | 0.983 | 0.980 |
| country | 377 | 0.985 | 0.985 | 0.979 | 0.981 |
| verbatim_elevation | 143 | 0.986 | 0.982 | 0.986 | 0.983 |
| exsiccata_title | 324 | 0.988 | 0.995 | 0.983 | 0.988 |
| catalog_number | 378 | 0.993 | 0.993 | 0.993 | 0.993 |

## CER, parse_rate, sec/img summary

| Run | parse_rate | mean sec/img | n_records |
|---|---|---|---|
| p4h_r32_lr2e4_ep5 | 0.9974 | 13.97 | 378 |
| p5j_r32_lr7e5_ep5_a256 | 0.9973 | 14.01 | 376 |
| p4g_r48_lr1e4_ep5 | 1.0000 | 13.94 | 386 |
| p4g_r16_lr1e4_ep8 | 1.0000 | 13.71 | 383 |
