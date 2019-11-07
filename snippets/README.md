# Teachable Machine export snippets

This folder contains markdown snippets that are being displayed inside the export panel inside [Teachable Machine](https://teachablemachine.withgoogle.com/). The snippets contain code and instructions on how to use the exported models from Teachable Machine in languages like Javascript, Java and Python. This section is split up into folders for each type of model - image, pose and audio, and within those folders split up into different model export types, like tensorflow js and tflite. 

If you want to contribute with fixes or new snippets, please feel free to submit a pull request! 

### Deployment
Each commit to master branch triggers [Cloud Build](https://pantheon.corp.google.com/cloud-build/dashboard?project=gweb-teachable-ai) task that copies the markdown files to a cloud storage bucket that is served to the Teachable Machine frontend.

[markdown/index.json](markdown/index.json) serves as the index file of all markdown files, and contains the titles and tooltips of each file. If a new file is being added, you should also add it to the index file. 
