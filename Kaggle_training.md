# Instruction on using Kaggle GPU

Author: qijunluo@link.cuhk.edu.cn

The file contains scripts for uploading your code to Kaggle to train. Best open in markdown viewer.

**This file is mainly written for groups who does not have enough computational resources, instructing them how to use the GPU resource offered by Kaggle platform.** If you already have a machine with powerful GPU (with at least 15G GPU RAM), then you don't need to read this file.
> To check your GPU utility, run `nvidia-smi` in the terminal.

After finishing your code, run the following commands in your terminal line-by-line **under the parent folder of `src` directory** (the `minChatGPT` folder by default). The code will be uploaded to Kaggle platform to be executed. For more information about the meaning of the kaggle API, see https://www.kaggle.com/docs/api#getting-started-installation-&-authentication.

**Basic workflow:** 

* **First time execution:** step 0 &rarr; step 1 &rarr; step 2 case 1 &rarr; step 3.
* **Otherwise:** step 1 &rarr; step 2 case 2 &rarr; step 3.

## Step 0: Set up
Install Kaggle Package
```bash
pip install kaggle
```
Go to [Kaggle](https://www.kaggle.com/) and register a account. Then, get authentication by
1. Go to the 'Account' tab of your user profile and select 'Create New Token'. This will trigger the download of kaggle.json, a file containing your API credentials.
2. Put the `kaggle.json` file under `~/.kaggle/` for Linux, OSX, and other UNIX-based operating systems, or `C:\Users\<Windows-username>\.kaggle\` for Windows system. If the token is not there, an error will be raised. Hence, once you’ve downloaded the token, you should move it from your Downloads folder to this folder.

For more information about authentication, see https://www.kaggle.com/docs/api#getting-started-installation-&-authentication.

## Step 1: Pack your code
Once you have finished your code in `./src`, run the following command to pack the code into folder `latest_code`.
```bash
rm -rf latest_code # if the folder already exists, remove it
rsync -av --exclude "runs" --exclude "*.json"  --exclude "__pycache__" --exclude ".git"  ./src ./latest_code # pack all the code into the latest_code folder. You may want to add excluded files here
```

## Step 2: Upload the code to Kaggle
### Case 1: First time execution.
Our code will be treated as a dataset in Kaggle. Later when executing the code in Kaggle, we will load the code by loading the dataset.
```bash
kaggle datasets init -p ./latest_code # generate a configuration file dataset-metadata.json under ./latest_code
```
Fill the marked place in `./latest_code/dataset-metadata.json` following the requirement of https://github.com/Kaggle/kaggle-api/wiki/Dataset-Metadata. You may refer to `dataset-metadata-template.json` for a template. Then, upload the code to Kaggle by

```bash
cp ./latest_code/dataset-metadata.json . # save the configuration file to current folder so that we don't need to execute it again
kaggle datasets create -p ./latest_code --dir-mode zip # upload the code to Kaggle
```

### Case 2: Update the existing code.
```bash
cp dataset-metadata.json ./latest_code/ # use existing configuration file
kaggle datasets version -p ./latest_code --dir-mode zip -m <your commit message> # you can optionally add commit message to help identify the code version.
```

## Step 3: Execute the code
If `./kernel-metadata.json` does not exist, run the following command to create one. Then, fill the marked place in `./kernel-metadata.json` following the requirement of https://github.com/Kaggle/kaggle-api/wiki/Kernel-Metadata; see `kernel-metadata-template.json` for a template. Then use the following command to submit the job to Kaggle.
```bash
kaggle kernels init -p . # generate a configuration file ./kernel-metadata.json for notebook execution.
```

Once `./kernel-metadata.json` is properly setup, run the following command to submit the job.
```bash
kaggle kernels push -p .
```

You can then view the execution using the link in the terminal output.

**Tips for more efficient workflow:** You may want to execute the notebook in Kaggle interactively instead of submiting the whole job for each time. In this way, you don't need to start a new kernel for each run, and thereby don't need to re-install the packages. To do this, find your notebook in Kaggle and click the `Edit` button. Note that you should select the necessary dataset manually now (i.e. the main-code dataset uploaded by yourself, and the finetuning-dataset. You can type MDS5210 to search for the finetuning-dataset.) Then, you only need to run step 2 to upload your main code, use `main.ipynb` to copy the code into working directory, and execute your latest code.