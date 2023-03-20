# layerdivider
A tool to divide a single illustration into a layered structure.
![スクリーンショット 2023-03-07 034638](https://user-images.githubusercontent.com/48423148/223202706-5c6e9108-0cf4-40dc-b840-9c3df3d540da.png)

![スクリーンショット_20230307_035053](https://user-images.githubusercontent.com/48423148/223345165-e4e2e7f6-059f-445a-ac3d-2c9c3ecd094a.png)


https://user-images.githubusercontent.com/48423148/223344286-bf2dff31-3fc5-4970-8d68-86274f1f36eb.mp4

# Install
## use Google Golab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mattyamonaca/layerdivider/blob/main/layerdivider_launch.ipynb)
1. Click colab button.
2. Run all cells.
3. Click output addres(Running on public URL: https: //xxxxxxx.gradio.live).


## Local Install
### Windows Installation
#### Required Dependencies
Python 3.10.8 and Git

#### install Step
1. 
``` 
git clone https://github.com/mattyamonaca/layerdivider
```

2. run `install.ps1` first time use, waiting for installation to complete.
3. run `run_gui.ps1` to open local gui.
4. open website localhost:port to use(The default is localhost:7860). 

#### Optional: For Python Launcher Users
If you installed Python launcher (py command), you can use another method for installation.
1. download this repository, for example by running
``` PowerShell
git clone https://github.com/mattyamonaca/layerdivider
```
2. run `install_with_launcher.ps1` instead of `install.ps1` to install dependent packages.
3. run `run_gui.ps1` to open local gui.
4. wait a few moments; then PowerShell outputs URL as below:
```PowerShell
Running on local URL: http://127.0.0.1:7860
```
Then open this URL by your browser.

# 処理内容
1. 入力された画像をピクセル単位でRGB情報に従いクラスタリング
2. 色の類似度（CIEDE2000基準）が近いクラスタを統合
3. 入力された画像をブラー処理で平滑化
4. クラスタごとにブラー処理後の色の平均値を出し、算出した平均値ですべてのピクセルを塗りなおし
5. 2-4を指定された回数繰り返し実行
6. 最終的なクラスタリング結果に基づき、ベースとなるレイヤーを作成
7. ベースレイヤーの各色を、入力された画像のクラスタ毎の平均色で塗りなおし
8. ベースレイヤーとオリジナルの色差に基づいて効果レイヤーを算出

# Processing content
1. Cluster the input image based on RGB information at the pixel level.
2. Merge clusters with similar color similarity (based on CIEDE2000 criteria).
3. Smooth the input image using a blur process.
4. For each cluster, calculate the average color value after blurring and repaint all pixels with this calculated value.
5. Repeat steps 2-4 for a specified number of times.
6. Create a base layer based on the final clustering result.
7. Repaint each color in the base layer with the average color of each cluster in the input image.
8. Calculate an effect layer based on differences between the base layer and original colors.

# パラメータ説明
* roop: 処理2-4を繰り返す回数
* init_cluster: 処理1で生成するクラスタの数（大きいほど細かくレイヤー分けされる）
* ciede_threshold: 処理2でどの程度色が類似していたらクラスタを結合するか決める閾値
* blur_size: 処理3でかけるブラー処理の大きさ（大きいほど強くぼかす）
* output_layer_mode
    * normal: 通常レイヤーのみで出力されるPSDを構成
    * composite: 通常レイヤー、スクリーンレイヤー、乗算レイヤー、減算レイヤー、加算レイヤーを組み合わせて出力されるPSDを構成

# Parameter Description
* roop: Number of times to repeat processing 2-4.
* init_cluster: Number of clusters generated in process 1 (the larger the number, the more finely layered it is).
* ciede_threshold: Threshold for determining when to combine clusters in process 2 based on how similar their colors are.
* blur_size: Size of the blur applied in process 3 (the larger the size, the stronger the blurring effect).
* output_layer_mode:
    * normal: Constructs a PSD that only includes normal layers.
    * composite: Constructs a PSD by combining normal layers with screen, multiply, subtract and add layers.
