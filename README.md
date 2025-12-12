# spring8_distribution

Distribution version for stitching, center of rotation correction, and reconstruction of projections captured at SPring-8 in December 2025.

## 環境構築

- Visual Studio Code
- MSVC(Visual StudioのC/C++用コンパイラ)
- CMake
- HDF5
- CUDA Toolkit

### Visual Studio Code

- インストールしてなかったら[こちらから](https://code.visualstudio.com/).
- 以下の拡張機能をインストール.
  - C/C++ Extension Pack  
    <img src= "images/image20251211-165216.png" alt= "" width="40%">

### MSVCのインストール

- Visual Studio がインストール済みなら多分入っている(以下確認手順).
  - Visual Studio Installerを開く  
  <img src= "images/image20251211-182120.png" alt= "" width="60%">  

  - 変更をクリック  
  <img src= "images/image20251211-182332.png" alt= "" width="60%">  

  - C++によるデスクトップ開発&rarr;VS **2022**のMSVCにチェックついてたら入ってる. ついてなかったら, チェック入れてインストール(2026ではダメ).  
  <img src= "images/image20251211-183025.png" alt= "" width="100%">  

- Visual Studio が入ってない場合は[こちらから](https://visualstudio.microsoft.com/ja/vs/older-downloads/) Build Tools for Visual Studio 2022 を入れるとよい. そのときもインストールする際に上記のツールを選択すればよい.

### CMakeのインストール

- [こちらから](https://cmake.org/download/)最新版を入れておけば問題ない
<img src= "images/image20251211-184103.png" alt= "" width="50%">  

(すでに入っている場合も3.31以降なら確実に動くはず)
- install options では <u>Add CMake to the system PATH for (好きなほう)</u> を選択

### HDF5のインストール

- [こちらから](https://github.com/HDFGroup/hdf5/releases/tag/2.0.0) hdf5-x.x.x-win-vs2022_cl.msiをダウンロードして実行  
<img src= "images/image20251211-185609.png" alt= "" width="50%">  

- 忘れたけど, PATHを追加できそうな設定があったらする.なかったら, インストール先のbinを環境変数のPATHに追加(自分の場合は、下の画像のところ). **インストール先はこの後の設定で必要になるので、確認しておくこと.**  
<img src= "images/image20251211-190603.png" alt= "" width="100%">  

- VS CodeでCMAKE_PREFIX_PATHの設定
  - VS Codeを開いて`ctrl + ,`でSettingsを開いて, 下の画像の黄矢印部をクリックして settings.jsonを開く.  
  <img src= "images/image20251211-193358.png" alt= "" width="100%">  

  - settings.jsonに以下を追加(2.0.0なら同じはず).
    ```settings.json
    "cmake.configureSettings": {
        "CMAKE_PREFIX_PATH": [
            "(インストール先)\\HDF_Group\\HDF5\\2.0.0\\cmake"
        ]
    },
    ```
    他の設定もあると思うので, 自分のsettings.jsonは以下のような感じ.
    <img src= "images/image20251211-194931.png" alt= "" width="100%">  

### CUDA Toolkitのインストール

- 自分のパソコンのGPUを調べる(タスクマネージャー開いたらわかる).
  <img src= "images/image20251211-195643.png" alt= "" width="80%">  

- [こちらから](https://www.nvidia.com/ja-jp/drivers/)対応するドライバをインストール  
  <img src= "images/image20251211-195802.png" alt= "" width="40%">  

    - Game Ready, StudioはどっちでもOK. そのまま進めてダウンロード
- [こちらから](https://developer.nvidia.com/cuda/toolkit) CUDA Toolkitをインストール
  - 環境に合わせて選択してDownload  
  <img src= "images/image20251211-200659.png" alt= "" width="100%">  

  - ダウンロードできたら, 開いてデフォルト設定のままインストール.

## 使い方
