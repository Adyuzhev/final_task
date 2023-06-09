#!/bin/bash
pip install dvc
PATH=$PATH:~/.local/bin
echo $PATH
dvc init
dvc remote add --default drive "gdrive://1wwnGLj2QkFiikQeoNsX5tZ5KVTMaLx0D"
dvc remote modify drive gdrive_acknowledge_abuse true
pip install dvc_gdrive
dvc remote modify drive gdrive_use_service_account true
dvc remote modify drive --local gdrive_service_account_json_file_path $1
dvc pull
