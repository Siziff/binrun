# Exp.5 VLM summary

| model                          | mode   |   records |   parse_rate |   micro_F1 |   macro_F1 |   soft_macro_R |   mean_runtime_s |
|:-------------------------------|:-------|----------:|-------------:|-----------:|-----------:|---------------:|-----------------:|
| Qwen/Qwen3-VL-30B-A3B-Instruct | A      |       314 |       0.9650 |     0.8079 |     0.6531 |         0.7257 |          49.0000 |
| Qwen/Qwen3-VL-8B-Instruct      | A      |       394 |       0.9822 |     0.8072 |     0.6725 |         0.7426 |          16.3600 |
| Qwen/Qwen3-VL-30B-A3B-Thinking | A      |        16 |       0.8750 |     0.6964 |     0.6397 |         0.6377 |         242.7900 |
| OpenGVLab/InternVL3-8B-hf      | A      |       372 |       0.9892 |     0.6677 |     0.5618 |         0.6482 |          15.3700 |
| Qwen/Qwen3-VL-8B-Thinking      | A      |       209 |       0.5646 |     0.5439 |     0.4521 |         0.4328 |         314.2900 |
| Qwen/Qwen3-VL-8B-Instruct      | B      |       394 |       0.9898 |     0.8075 |     0.6712 |         0.7560 |          13.9900 |
| Qwen/Qwen3-VL-30B-A3B-Instruct | B      |       394 |       0.9848 |     0.7983 |     0.6420 |         0.7239 |          33.0200 |
| OpenGVLab/InternVL3-8B-hf      | B      |       326 |       0.9969 |     0.6826 |     0.5640 |         0.6330 |           5.2900 |
| Qwen/Qwen3-VL-30B-A3B-Thinking | B      |        24 |       0.5000 |     0.5432 |     0.4605 |         0.3546 |         153.5600 |
