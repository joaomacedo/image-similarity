sudo docker run -it -p 8888:8888 -v /home/jm:/home_docker ufoym/deepo:all-jupyter-py36-cpu jupyter notebook --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token= --notebook-dir='/home_docker/report_similaridade'
