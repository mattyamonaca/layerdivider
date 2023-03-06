# layerdivider
A tool to divide a single illustration into a layered structure.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mattyamonaca/layerdivider/blob/main/layerdivider_launch.ipynb)

![スクリーンショット 2023-03-07 034638](https://user-images.githubusercontent.com/48423148/223202706-5c6e9108-0cf4-40dc-b840-9c3df3d540da.png)

![スクリーンショット 2023-03-07 035056](https://user-images.githubusercontent.com/48423148/223203504-f443f7a7-4123-42e4-b0fb-cabde491712a.png)


# 処理内容
1. 入力された画像をピクセル単位でRGB情報に従いクラスタリング
2. 色の類似度（CIEDE2000基準）が近いクラスタを統合
3. 入力された画像をブラー処理で平滑化
4. クラスタごとにブラー処理後の色の平均値を出し、算出した平均値ですべてのピクセルを塗りなおし
5. 2-4を指定された回数繰り返し実行
6. 最終的なクラスタリング結果に基づき、ベースとなるレイヤーを作成
7. ベースレイヤーの各色を、入力された画像のクラスタ毎の平均色で塗りなおし
8. ベースレイヤーとオリジナルの色差に基づいて効果レイヤーを算出

# パラメータ説明
* roop: 処理2-4を繰り返す回数
* init_cluster: 処理1で生成するクラスタの数（大きいほど細かくレイヤー分けされる）
* ciede_threshold: 処理2でどの程度色が類似していたらクラスタを結合するか決める閾値
* blur_size: 処理3でかけるブラー処理の大きさ（大きいほど強くぼかす）
* output_layer_mode
    * normal: 通常レイヤーのみで出力されるPSDを構成
    * composite: 通常レイヤー、スクリーンレイヤー、乗算レイヤー、減算レイヤー、加算レイヤーを組み合わせて出力されるPSDを構成
