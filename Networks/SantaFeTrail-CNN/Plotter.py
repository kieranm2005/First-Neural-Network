import matplotlib.pyplot as plt
import numpy as np
import re

data = """
Episode 1, Total Reward: 6, Epsilon: 0.995
Episode 2, Total Reward: 17, Epsilon: 0.990
Episode 3, Total Reward: 13, Epsilon: 0.985
Episode 4, Total Reward: 14, Epsilon: 0.980
Episode 5, Total Reward: 12, Epsilon: 0.975
Episode 6, Total Reward: 9, Epsilon: 0.970
Episode 7, Total Reward: 13, Epsilon: 0.966
Episode 8, Total Reward: 13, Epsilon: 0.961
Episode 9, Total Reward: 14, Epsilon: 0.956
Episode 10, Total Reward: 12, Epsilon: 0.951
Episode 11, Total Reward: 10, Epsilon: 0.946
Episode 12, Total Reward: 11, Epsilon: 0.942
Episode 13, Total Reward: 12, Epsilon: 0.937
Episode 14, Total Reward: 11, Epsilon: 0.932
Episode 15, Total Reward: 15, Epsilon: 0.928
Episode 16, Total Reward: 17, Epsilon: 0.923
Episode 17, Total Reward: 17, Epsilon: 0.918
Episode 18, Total Reward: 12, Epsilon: 0.914
Episode 19, Total Reward: 16, Epsilon: 0.909
Episode 20, Total Reward: 19, Epsilon: 0.905
Episode 21, Total Reward: 19, Epsilon: 0.900
Episode 22, Total Reward: 19, Epsilon: 0.896
Episode 23, Total Reward: 26, Epsilon: 0.891
Episode 24, Total Reward: 15, Epsilon: 0.887
Episode 25, Total Reward: 14, Epsilon: 0.882
Episode 26, Total Reward: 10, Epsilon: 0.878
Episode 27, Total Reward: 22, Epsilon: 0.873
Episode 28, Total Reward: 25, Epsilon: 0.869
Episode 29, Total Reward: 18, Epsilon: 0.865
Episode 30, Total Reward: 23, Epsilon: 0.860
Episode 31, Total Reward: 22, Epsilon: 0.856
Episode 32, Total Reward: 19, Epsilon: 0.852
Episode 33, Total Reward: 29, Epsilon: 0.848
Episode 34, Total Reward: 19, Epsilon: 0.843
Episode 35, Total Reward: 14, Epsilon: 0.839
Episode 36, Total Reward: 18, Epsilon: 0.835
Episode 37, Total Reward: 27, Epsilon: 0.831
Episode 38, Total Reward: 13, Epsilon: 0.827
Episode 39, Total Reward: 16, Epsilon: 0.822
Episode 40, Total Reward: 18, Epsilon: 0.818
Episode 41, Total Reward: 17, Epsilon: 0.814
Episode 42, Total Reward: 23, Epsilon: 0.810
Episode 43, Total Reward: 20, Epsilon: 0.806
Episode 44, Total Reward: 20, Epsilon: 0.802
Episode 45, Total Reward: 17, Epsilon: 0.798
Episode 46, Total Reward: 24, Epsilon: 0.794
Episode 47, Total Reward: 22, Epsilon: 0.790
Episode 48, Total Reward: 20, Epsilon: 0.786
Episode 49, Total Reward: 17, Epsilon: 0.782
Episode 50, Total Reward: 14, Epsilon: 0.778
Episode 51, Total Reward: 22, Epsilon: 0.774
Episode 52, Total Reward: 15, Epsilon: 0.771
Episode 53, Total Reward: 13, Epsilon: 0.767
Episode 54, Total Reward: 24, Epsilon: 0.763
Episode 55, Total Reward: 15, Epsilon: 0.759
Episode 56, Total Reward: 18, Epsilon: 0.755
Episode 57, Total Reward: 27, Epsilon: 0.751
Episode 58, Total Reward: 13, Epsilon: 0.748
Episode 59, Total Reward: 17, Epsilon: 0.744
Episode 60, Total Reward: 29, Epsilon: 0.740
Episode 61, Total Reward: 16, Epsilon: 0.737
Episode 62, Total Reward: 23, Epsilon: 0.733
Episode 63, Total Reward: 20, Epsilon: 0.729
Episode 64, Total Reward: 17, Epsilon: 0.726
Episode 65, Total Reward: 31, Epsilon: 0.722
Episode 66, Total Reward: 20, Epsilon: 0.718
Episode 67, Total Reward: 9, Epsilon: 0.715
Episode 68, Total Reward: 14, Epsilon: 0.711
Episode 69, Total Reward: 19, Epsilon: 0.708
Episode 70, Total Reward: 22, Epsilon: 0.704
Episode 71, Total Reward: 28, Epsilon: 0.701
Episode 72, Total Reward: 27, Epsilon: 0.697
Episode 73, Total Reward: 36, Epsilon: 0.694
Episode 74, Total Reward: 22, Epsilon: 0.690
Episode 75, Total Reward: 30, Epsilon: 0.687
Episode 76, Total Reward: 26, Epsilon: 0.683
Episode 77, Total Reward: 23, Epsilon: 0.680
Episode 78, Total Reward: 19, Epsilon: 0.676
Episode 79, Total Reward: 34, Epsilon: 0.673
Episode 80, Total Reward: 29, Epsilon: 0.670
Episode 81, Total Reward: 37, Epsilon: 0.666
Episode 82, Total Reward: 25, Epsilon: 0.663
Episode 83, Total Reward: 25, Epsilon: 0.660
Episode 84, Total Reward: 35, Epsilon: 0.656
Episode 85, Total Reward: 41, Epsilon: 0.653
Episode 86, Total Reward: 27, Epsilon: 0.650
Episode 87, Total Reward: 27, Epsilon: 0.647
Episode 88, Total Reward: 40, Epsilon: 0.643
Episode 89, Total Reward: 28, Epsilon: 0.640
Episode 90, Total Reward: 32, Epsilon: 0.637
Episode 91, Total Reward: 33, Epsilon: 0.634
Episode 92, Total Reward: 25, Epsilon: 0.631
Episode 93, Total Reward: 36, Epsilon: 0.627
Episode 94, Total Reward: 41, Epsilon: 0.624
Episode 95, Total Reward: 38, Epsilon: 0.621
Episode 96, Total Reward: 29, Epsilon: 0.618
Episode 97, Total Reward: 43, Epsilon: 0.615
Episode 98, Total Reward: 34, Epsilon: 0.612
Episode 99, Total Reward: 37, Epsilon: 0.609
Episode 100, Total Reward: 26, Epsilon: 0.606
Episode 101, Total Reward: 36, Epsilon: 0.603
Episode 102, Total Reward: 40, Epsilon: 0.600
Episode 103, Total Reward: 40, Epsilon: 0.597
Episode 104, Total Reward: 30, Epsilon: 0.594
Episode 105, Total Reward: 32, Epsilon: 0.591
Episode 106, Total Reward: 29, Epsilon: 0.588
Episode 107, Total Reward: 34, Epsilon: 0.585
Episode 108, Total Reward: 40, Epsilon: 0.582
Episode 109, Total Reward: 29, Epsilon: 0.579
Episode 110, Total Reward: 31, Epsilon: 0.576
Episode 111, Total Reward: 38, Epsilon: 0.573
Episode 112, Total Reward: 25, Epsilon: 0.570
Episode 113, Total Reward: 31, Epsilon: 0.568
Episode 114, Total Reward: 43, Epsilon: 0.565
Episode 115, Total Reward: 21, Epsilon: 0.562
Episode 116, Total Reward: 25, Epsilon: 0.559
Episode 117, Total Reward: 34, Epsilon: 0.556
Episode 118, Total Reward: 40, Epsilon: 0.554
Episode 119, Total Reward: 36, Epsilon: 0.551
Episode 120, Total Reward: 44, Epsilon: 0.548
Episode 121, Total Reward: 49, Epsilon: 0.545
Episode 122, Total Reward: 45, Epsilon: 0.543
Episode 123, Total Reward: 40, Epsilon: 0.540
Episode 124, Total Reward: 29, Epsilon: 0.537
Episode 125, Total Reward: 47, Epsilon: 0.534
Episode 126, Total Reward: 46, Epsilon: 0.532
Episode 127, Total Reward: 45, Epsilon: 0.529
Episode 128, Total Reward: 40, Epsilon: 0.526
Episode 129, Total Reward: 38, Epsilon: 0.524
Episode 130, Total Reward: 42, Epsilon: 0.521
Episode 131, Total Reward: 54, Epsilon: 0.519
Episode 132, Total Reward: 45, Epsilon: 0.516
Episode 133, Total Reward: 44, Epsilon: 0.513
Episode 134, Total Reward: 43, Epsilon: 0.511
Episode 135, Total Reward: 59, Epsilon: 0.508
Episode 136, Total Reward: 52, Epsilon: 0.506
Episode 137, Total Reward: 43, Epsilon: 0.503
Episode 138, Total Reward: 49, Epsilon: 0.501
Episode 139, Total Reward: 43, Epsilon: 0.498
Episode 140, Total Reward: 39, Epsilon: 0.496
Episode 141, Total Reward: 44, Epsilon: 0.493
Episode 142, Total Reward: 39, Epsilon: 0.491
Episode 143, Total Reward: 42, Epsilon: 0.488
Episode 144, Total Reward: 42, Epsilon: 0.486
Episode 145, Total Reward: 48, Epsilon: 0.483
Episode 146, Total Reward: 50, Epsilon: 0.481
Episode 147, Total Reward: 43, Epsilon: 0.479
Episode 148, Total Reward: 45, Epsilon: 0.476
Episode 149, Total Reward: 44, Epsilon: 0.474
Episode 150, Total Reward: 47, Epsilon: 0.471
Episode 151, Total Reward: 45, Epsilon: 0.469
Episode 152, Total Reward: 40, Epsilon: 0.467
Episode 153, Total Reward: 39, Epsilon: 0.464
Episode 154, Total Reward: 50, Epsilon: 0.462
Episode 155, Total Reward: 49, Epsilon: 0.460
Episode 156, Total Reward: 58, Epsilon: 0.458
Episode 157, Total Reward: 46, Epsilon: 0.455
Episode 158, Total Reward: 57, Epsilon: 0.453
Episode 159, Total Reward: 42, Epsilon: 0.451
Episode 160, Total Reward: 40, Epsilon: 0.448
Episode 161, Total Reward: 46, Epsilon: 0.446
Episode 162, Total Reward: 45, Epsilon: 0.444
Episode 163, Total Reward: 38, Epsilon: 0.442
Episode 164, Total Reward: 46, Epsilon: 0.440
Episode 165, Total Reward: 45, Epsilon: 0.437
Episode 166, Total Reward: 55, Epsilon: 0.435
Episode 167, Total Reward: 47, Epsilon: 0.433
Episode 168, Total Reward: 50, Epsilon: 0.431
Episode 169, Total Reward: 50, Epsilon: 0.429
Episode 170, Total Reward: 49, Epsilon: 0.427
Episode 171, Total Reward: 46, Epsilon: 0.424
Episode 172, Total Reward: 43, Epsilon: 0.422
Episode 173, Total Reward: 48, Epsilon: 0.420
Episode 174, Total Reward: 44, Epsilon: 0.418
Episode 175, Total Reward: 50, Epsilon: 0.416
Episode 176, Total Reward: 50, Epsilon: 0.414
Episode 177, Total Reward: 49, Epsilon: 0.412
Episode 178, Total Reward: 49, Epsilon: 0.410
Episode 179, Total Reward: 50, Epsilon: 0.408
Episode 180, Total Reward: 50, Epsilon: 0.406
Episode 181, Total Reward: 47, Epsilon: 0.404
Episode 182, Total Reward: 47, Epsilon: 0.402
Episode 183, Total Reward: 47, Epsilon: 0.400
Episode 184, Total Reward: 46, Epsilon: 0.398
Episode 185, Total Reward: 46, Epsilon: 0.396
Episode 186, Total Reward: 45, Epsilon: 0.394
Episode 187, Total Reward: 46, Epsilon: 0.392
Episode 188, Total Reward: 52, Epsilon: 0.390
Episode 189, Total Reward: 53, Epsilon: 0.388
Episode 190, Total Reward: 52, Epsilon: 0.386
Episode 191, Total Reward: 50, Epsilon: 0.384
Episode 192, Total Reward: 48, Epsilon: 0.382
Episode 193, Total Reward: 56, Epsilon: 0.380
Episode 194, Total Reward: 55, Epsilon: 0.378
Episode 195, Total Reward: 52, Epsilon: 0.376
Episode 196, Total Reward: 56, Epsilon: 0.374
Episode 197, Total Reward: 56, Epsilon: 0.373
Episode 198, Total Reward: 51, Epsilon: 0.371
Episode 199, Total Reward: 49, Epsilon: 0.369
Episode 200, Total Reward: 57, Epsilon: 0.367
Episode 201, Total Reward: 49, Epsilon: 0.365
Episode 202, Total Reward: 55, Epsilon: 0.363
Episode 203, Total Reward: 63, Epsilon: 0.361
Episode 204, Total Reward: 51, Epsilon: 0.360
Episode 205, Total Reward: 48, Epsilon: 0.358
Episode 206, Total Reward: 46, Epsilon: 0.356
Episode 207, Total Reward: 59, Epsilon: 0.354
Episode 208, Total Reward: 58, Epsilon: 0.353
Episode 209, Total Reward: 56, Epsilon: 0.351
Episode 210, Total Reward: 65, Epsilon: 0.349
Episode 211, Total Reward: 64, Epsilon: 0.347
Episode 212, Total Reward: 51, Epsilon: 0.346
Episode 213, Total Reward: 59, Epsilon: 0.344
Episode 214, Total Reward: 57, Epsilon: 0.342
Episode 215, Total Reward: 63, Epsilon: 0.340
Episode 216, Total Reward: 54, Epsilon: 0.339
Episode 217, Total Reward: 45, Epsilon: 0.337
Episode 218, Total Reward: 60, Epsilon: 0.335
Episode 219, Total Reward: 59, Epsilon: 0.334
Episode 220, Total Reward: 55, Epsilon: 0.332
Episode 221, Total Reward: 52, Epsilon: 0.330
Episode 222, Total Reward: 49, Epsilon: 0.329
Episode 223, Total Reward: 51, Epsilon: 0.327
Episode 224, Total Reward: 51, Epsilon: 0.325
Episode 225, Total Reward: 64, Epsilon: 0.324
Episode 226, Total Reward: 59, Epsilon: 0.322
Episode 227, Total Reward: 64, Epsilon: 0.321
Episode 228, Total Reward: 63, Epsilon: 0.319
Episode 229, Total Reward: 63, Epsilon: 0.317
Episode 230, Total Reward: 65, Epsilon: 0.316
Episode 231, Total Reward: 61, Epsilon: 0.314
Episode 232, Total Reward: 58, Epsilon: 0.313
Episode 233, Total Reward: 62, Epsilon: 0.311
Episode 234, Total Reward: 63, Epsilon: 0.309
Episode 235, Total Reward: 60, Epsilon: 0.308
Episode 236, Total Reward: 39, Epsilon: 0.306
Episode 237, Total Reward: 69, Epsilon: 0.305
Episode 238, Total Reward: 66, Epsilon: 0.303
Episode 239, Total Reward: 62, Epsilon: 0.302
Episode 240, Total Reward: 64, Epsilon: 0.300
Episode 241, Total Reward: 63, Epsilon: 0.299
Episode 242, Total Reward: 65, Epsilon: 0.297
Episode 243, Total Reward: 60, Epsilon: 0.296
Episode 244, Total Reward: 65, Epsilon: 0.294
Episode 245, Total Reward: 66, Epsilon: 0.293
Episode 246, Total Reward: 64, Epsilon: 0.291
Episode 247, Total Reward: 65, Epsilon: 0.290
Episode 248, Total Reward: 65, Epsilon: 0.288
Episode 249, Total Reward: 65, Epsilon: 0.287
Episode 250, Total Reward: 63, Epsilon: 0.286
Episode 251, Total Reward: 77, Epsilon: 0.284
Episode 252, Total Reward: 67, Epsilon: 0.283
Episode 253, Total Reward: 64, Epsilon: 0.281
Episode 254, Total Reward: 66, Epsilon: 0.280
Episode 255, Total Reward: 56, Epsilon: 0.279
Episode 256, Total Reward: 63, Epsilon: 0.277
Episode 257, Total Reward: 60, Epsilon: 0.276
Episode 258, Total Reward: 65, Epsilon: 0.274
Episode 259, Total Reward: 65, Epsilon: 0.273
Episode 260, Total Reward: 58, Epsilon: 0.272
Episode 261, Total Reward: 65, Epsilon: 0.270
Episode 262, Total Reward: 58, Epsilon: 0.269
Episode 263, Total Reward: 63, Epsilon: 0.268
Episode 264, Total Reward: 70, Epsilon: 0.266
Episode 265, Total Reward: 77, Epsilon: 0.265
Episode 266, Total Reward: 61, Epsilon: 0.264
Episode 267, Total Reward: 60, Epsilon: 0.262
Episode 268, Total Reward: 66, Epsilon: 0.261
Episode 269, Total Reward: 68, Epsilon: 0.260
Episode 270, Total Reward: 68, Epsilon: 0.258
Episode 271, Total Reward: 67, Epsilon: 0.257
Episode 272, Total Reward: 62, Epsilon: 0.256
Episode 273, Total Reward: 57, Epsilon: 0.255
Episode 274, Total Reward: 64, Epsilon: 0.253
Episode 275, Total Reward: 65, Epsilon: 0.252
Episode 276, Total Reward: 68, Epsilon: 0.251
Episode 277, Total Reward: 62, Epsilon: 0.249
Episode 278, Total Reward: 69, Epsilon: 0.248
Episode 279, Total Reward: 66, Epsilon: 0.247
Episode 280, Total Reward: 69, Epsilon: 0.246
Episode 281, Total Reward: 69, Epsilon: 0.245
Episode 282, Total Reward: 67, Epsilon: 0.243
Episode 283, Total Reward: 67, Epsilon: 0.242
Episode 284, Total Reward: 70, Epsilon: 0.241
Episode 285, Total Reward: 68, Epsilon: 0.240
Episode 286, Total Reward: 73, Epsilon: 0.238
Episode 287, Total Reward: 65, Epsilon: 0.237
Episode 288, Total Reward: 66, Epsilon: 0.236
Episode 289, Total Reward: 67, Epsilon: 0.235
Episode 290, Total Reward: 63, Epsilon: 0.234
Episode 291, Total Reward: 68, Epsilon: 0.233
Episode 292, Total Reward: 69, Epsilon: 0.231
Episode 293, Total Reward: 68, Epsilon: 0.230
Episode 294, Total Reward: 62, Epsilon: 0.229
Episode 295, Total Reward: 68, Epsilon: 0.228
Episode 296, Total Reward: 60, Epsilon: 0.227
Episode 297, Total Reward: 70, Epsilon: 0.226
Episode 298, Total Reward: 66, Epsilon: 0.225
Episode 299, Total Reward: 65, Epsilon: 0.223
Episode 300, Total Reward: 69, Epsilon: 0.222
Episode 301, Total Reward: 66, Epsilon: 0.221
Episode 302, Total Reward: 66, Epsilon: 0.220
Episode 303, Total Reward: 69, Epsilon: 0.219
Episode 304, Total Reward: 68, Epsilon: 0.218
Episode 305, Total Reward: 69, Epsilon: 0.217
Episode 306, Total Reward: 74, Epsilon: 0.216
Episode 307, Total Reward: 67, Epsilon: 0.215
Episode 308, Total Reward: 69, Epsilon: 0.214
Episode 309, Total Reward: 67, Epsilon: 0.212
Episode 310, Total Reward: 68, Epsilon: 0.211
Episode 311, Total Reward: 69, Epsilon: 0.210
Episode 312, Total Reward: 69, Epsilon: 0.209
Episode 313, Total Reward: 68, Epsilon: 0.208
Episode 314, Total Reward: 68, Epsilon: 0.207
Episode 315, Total Reward: 68, Epsilon: 0.206
Episode 316, Total Reward: 70, Epsilon: 0.205
Episode 317, Total Reward: 68, Epsilon: 0.204
Episode 318, Total Reward: 71, Epsilon: 0.203
Episode 319, Total Reward: 68, Epsilon: 0.202
Episode 320, Total Reward: 68, Epsilon: 0.201
Episode 321, Total Reward: 69, Epsilon: 0.200
Episode 322, Total Reward: 66, Epsilon: 0.199
Episode 323, Total Reward: 56, Epsilon: 0.198
Episode 324, Total Reward: 69, Epsilon: 0.197
Episode 325, Total Reward: 69, Epsilon: 0.196
Episode 326, Total Reward: 69, Epsilon: 0.195
Episode 327, Total Reward: 59, Epsilon: 0.194
Episode 328, Total Reward: 67, Epsilon: 0.193
Episode 329, Total Reward: 69, Epsilon: 0.192
Episode 330, Total Reward: 70, Epsilon: 0.191
Episode 331, Total Reward: 69, Epsilon: 0.190
Episode 332, Total Reward: 84, Epsilon: 0.189
Episode 333, Total Reward: 69, Epsilon: 0.188
Episode 334, Total Reward: 70, Epsilon: 0.187
Episode 335, Total Reward: 57, Epsilon: 0.187
Episode 336, Total Reward: 70, Epsilon: 0.186
Episode 337, Total Reward: 71, Epsilon: 0.185
Episode 338, Total Reward: 69, Epsilon: 0.184
Episode 339, Total Reward: 70, Epsilon: 0.183
Episode 340, Total Reward: 75, Epsilon: 0.182
Episode 341, Total Reward: 70, Epsilon: 0.181
Episode 342, Total Reward: 71, Epsilon: 0.180
Episode 343, Total Reward: 70, Epsilon: 0.179
Episode 344, Total Reward: 71, Epsilon: 0.178
Episode 345, Total Reward: 77, Epsilon: 0.177
Episode 346, Total Reward: 69, Epsilon: 0.177
Episode 347, Total Reward: 65, Epsilon: 0.176
Episode 348, Total Reward: 68, Epsilon: 0.175
Episode 349, Total Reward: 60, Epsilon: 0.174
Episode 350, Total Reward: 69, Epsilon: 0.173
Episode 351, Total Reward: 68, Epsilon: 0.172
Episode 352, Total Reward: 67, Epsilon: 0.171
Episode 353, Total Reward: 70, Epsilon: 0.170
Episode 354, Total Reward: 70, Epsilon: 0.170
Episode 355, Total Reward: 71, Epsilon: 0.169
Episode 356, Total Reward: 51, Epsilon: 0.168
Episode 357, Total Reward: 65, Epsilon: 0.167
Episode 358, Total Reward: 71, Epsilon: 0.166
Episode 359, Total Reward: 76, Epsilon: 0.165
Episode 360, Total Reward: 71, Epsilon: 0.165
Episode 361, Total Reward: 71, Epsilon: 0.164
Episode 362, Total Reward: 69, Epsilon: 0.163
Episode 363, Total Reward: 71, Epsilon: 0.162
Episode 364, Total Reward: 71, Epsilon: 0.161
Episode 365, Total Reward: 81, Epsilon: 0.160
Episode 366, Total Reward: 54, Epsilon: 0.160
Episode 367, Total Reward: 72, Epsilon: 0.159
Episode 368, Total Reward: 74, Epsilon: 0.158
Episode 369, Total Reward: 71, Epsilon: 0.157
Episode 370, Total Reward: 70, Epsilon: 0.157
Episode 371, Total Reward: 70, Epsilon: 0.156
Episode 372, Total Reward: 71, Epsilon: 0.155
Episode 373, Total Reward: 70, Epsilon: 0.154
Episode 374, Total Reward: 75, Epsilon: 0.153
Episode 375, Total Reward: 69, Epsilon: 0.153
Episode 376, Total Reward: 65, Epsilon: 0.152
Episode 377, Total Reward: 70, Epsilon: 0.151
Episode 378, Total Reward: 71, Epsilon: 0.150
Episode 379, Total Reward: 76, Epsilon: 0.150
Episode 380, Total Reward: 65, Epsilon: 0.149
Episode 381, Total Reward: 79, Epsilon: 0.148
Episode 382, Total Reward: 69, Epsilon: 0.147
Episode 383, Total Reward: 71, Epsilon: 0.147
Episode 384, Total Reward: 71, Epsilon: 0.146
Episode 385, Total Reward: 77, Epsilon: 0.145
Episode 386, Total Reward: 76, Epsilon: 0.144
Episode 387, Total Reward: 82, Epsilon: 0.144
Episode 388, Total Reward: 84, Epsilon: 0.143
Episode 389, Total Reward: 68, Epsilon: 0.142
Episode 390, Total Reward: 83, Epsilon: 0.142
Episode 391, Total Reward: 60, Epsilon: 0.141
Episode 392, Total Reward: 69, Epsilon: 0.140
Episode 393, Total Reward: 77, Epsilon: 0.139
Episode 394, Total Reward: 73, Epsilon: 0.139
Episode 395, Total Reward: 82, Epsilon: 0.138
Episode 396, Total Reward: 69, Epsilon: 0.137
Episode 397, Total Reward: 48, Epsilon: 0.137
Episode 398, Total Reward: 67, Epsilon: 0.136
Episode 399, Total Reward: 67, Epsilon: 0.135
Episode 400, Total Reward: 66, Epsilon: 0.135
Episode 401, Total Reward: 75, Epsilon: 0.134
Episode 402, Total Reward: 65, Epsilon: 0.133
Episode 403, Total Reward: 60, Epsilon: 0.133
Episode 404, Total Reward: 59, Epsilon: 0.132
Episode 405, Total Reward: 71, Epsilon: 0.131
Episode 406, Total Reward: 82, Epsilon: 0.131
Episode 407, Total Reward: 79, Epsilon: 0.130
Episode 408, Total Reward: 75, Epsilon: 0.129
Episode 409, Total Reward: 78, Epsilon: 0.129
Episode 410, Total Reward: 67, Epsilon: 0.128
Episode 411, Total Reward: 54, Epsilon: 0.127
Episode 412, Total Reward: 79, Epsilon: 0.127
Episode 413, Total Reward: 67, Epsilon: 0.126
Episode 414, Total Reward: 51, Epsilon: 0.126
Episode 415, Total Reward: 71, Epsilon: 0.125
Episode 416, Total Reward: 71, Epsilon: 0.124
Episode 417, Total Reward: 71, Epsilon: 0.124
Episode 418, Total Reward: 71, Epsilon: 0.123
Episode 419, Total Reward: 43, Epsilon: 0.122
Episode 420, Total Reward: 71, Epsilon: 0.122
Episode 421, Total Reward: 69, Epsilon: 0.121
Episode 422, Total Reward: 70, Epsilon: 0.121
Episode 423, Total Reward: 71, Epsilon: 0.120
Episode 424, Total Reward: 69, Epsilon: 0.119
Episode 425, Total Reward: 69, Epsilon: 0.119
Episode 426, Total Reward: 71, Epsilon: 0.118
Episode 427, Total Reward: 56, Epsilon: 0.118
Episode 428, Total Reward: 64, Epsilon: 0.117
Episode 429, Total Reward: 63, Epsilon: 0.116
Episode 430, Total Reward: 60, Epsilon: 0.116
Episode 431, Total Reward: 67, Epsilon: 0.115
Episode 432, Total Reward: 67, Epsilon: 0.115
Episode 433, Total Reward: 69, Epsilon: 0.114
Episode 434, Total Reward: 69, Epsilon: 0.114
Episode 435, Total Reward: 71, Epsilon: 0.113
Episode 436, Total Reward: 69, Epsilon: 0.112
Episode 437, Total Reward: 66, Epsilon: 0.112
Episode 438, Total Reward: 63, Epsilon: 0.111
Episode 439, Total Reward: 71, Epsilon: 0.111
Episode 440, Total Reward: 71, Epsilon: 0.110
Episode 441, Total Reward: 68, Epsilon: 0.110
Episode 442, Total Reward: 67, Epsilon: 0.109
Episode 443, Total Reward: 61, Epsilon: 0.109
Episode 444, Total Reward: 80, Epsilon: 0.108
Episode 445, Total Reward: 54, Epsilon: 0.107
Episode 446, Total Reward: 70, Epsilon: 0.107
Episode 447, Total Reward: 57, Epsilon: 0.106
Episode 448, Total Reward: 71, Epsilon: 0.106
Episode 449, Total Reward: 71, Epsilon: 0.105
Episode 450, Total Reward: 70, Epsilon: 0.105
Episode 451, Total Reward: 71, Epsilon: 0.104
Episode 452, Total Reward: 70, Epsilon: 0.104
Episode 453, Total Reward: 24, Epsilon: 0.103
Episode 454, Total Reward: 71, Epsilon: 0.103
Episode 455, Total Reward: 71, Epsilon: 0.102
Episode 456, Total Reward: 62, Epsilon: 0.102
Episode 457, Total Reward: 71, Epsilon: 0.101
Episode 458, Total Reward: 70, Epsilon: 0.101
Episode 459, Total Reward: 70, Epsilon: 0.100
Episode 460, Total Reward: 70, Epsilon: 0.100
Episode 461, Total Reward: 71, Epsilon: 0.100
Episode 462, Total Reward: 54, Epsilon: 0.100
Episode 463, Total Reward: 71, Epsilon: 0.100
Episode 464, Total Reward: 71, Epsilon: 0.100
Episode 465, Total Reward: 65, Epsilon: 0.100
Episode 466, Total Reward: 66, Epsilon: 0.100
Episode 467, Total Reward: 71, Epsilon: 0.100
Episode 468, Total Reward: 71, Epsilon: 0.100
Episode 469, Total Reward: 70, Epsilon: 0.100
Episode 470, Total Reward: 71, Epsilon: 0.100
Episode 471, Total Reward: 71, Epsilon: 0.100
Episode 472, Total Reward: 71, Epsilon: 0.100
Episode 473, Total Reward: 70, Epsilon: 0.100
Episode 474, Total Reward: 71, Epsilon: 0.100
Episode 475, Total Reward: 71, Epsilon: 0.100
Episode 476, Total Reward: 59, Epsilon: 0.100
Episode 477, Total Reward: 70, Epsilon: 0.100
Episode 478, Total Reward: 70, Epsilon: 0.100
Episode 479, Total Reward: 69, Epsilon: 0.100
Episode 480, Total Reward: 70, Epsilon: 0.100
Episode 481, Total Reward: 71, Epsilon: 0.100
Episode 482, Total Reward: 70, Epsilon: 0.100
Episode 483, Total Reward: 69, Epsilon: 0.100
Episode 484, Total Reward: 69, Epsilon: 0.100
Episode 485, Total Reward: 67, Epsilon: 0.100
Episode 486, Total Reward: 69, Epsilon: 0.100
Episode 487, Total Reward: 74, Epsilon: 0.100
Episode 488, Total Reward: 65, Epsilon: 0.100
Episode 489, Total Reward: 69, Epsilon: 0.100
Episode 490, Total Reward: 69, Epsilon: 0.100
Episode 491, Total Reward: 69, Epsilon: 0.100
Episode 492, Total Reward: 69, Epsilon: 0.100
Episode 493, Total Reward: 69, Epsilon: 0.100
Episode 494, Total Reward: 69, Epsilon: 0.100
Episode 495, Total Reward: 68, Epsilon: 0.100
Episode 496, Total Reward: 61, Epsilon: 0.100
Episode 497, Total Reward: 71, Epsilon: 0.100
Episode 498, Total Reward: 71, Epsilon: 0.100
Episode 499, Total Reward: 70, Epsilon: 0.100
Episode 500, Total Reward: 62, Epsilon: 0.100
"""

episodes = []
total_rewards = []
epsilons = []

for line in data.strip().split('\n'):
    match = re.match(r"Episode (\d+), Total Reward: (\d+), Epsilon: ([\d.]+)", line)
    if match:
        episodes.append(int(match.group(1)))
        total_rewards.append(int(match.group(2)))
        epsilons.append(float(match.group(3)))

# print("Episodes:", episodes)
# print("Total Rewards:", total_rewards)
# print("Epsilon values:", epsilons)

plt.plot(episodes, total_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Total Reward per Episode")
plt.show()

# Subsample the data to reduce the number of points
subsample_rate = 10
subsampled_episodes = episodes[::subsample_rate]
subsampled_rewards = total_rewards[::subsample_rate]
subsampled_epsilons = epsilons[::subsample_rate]
plt.plot(subsampled_episodes, subsampled_rewards, label='Total Reward')
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Subsampled Total Reward per Episode")
plt.legend()
plt.show()

# Print 10 best episodes based on total reward
best_episodes = sorted(zip(episodes, total_rewards, epsilons), key=lambda x: x[1], reverse=True)[:10]
print("Best 10 Episodes based on Total Reward:")
for episode, reward, epsilon in best_episodes:
    print(f"Episode {episode}, Total Reward: {reward}, Epsilon: {epsilon:.3f}")

def get_rewards():
    """Return the list of total rewards for all episodes (for import by other scripts)."""
    return total_rewards

def get_episodes():
    """Return the list of episode numbers."""
    return episodes

def get_epsilons():
    """Return the list of epsilon values."""
    return epsilons
