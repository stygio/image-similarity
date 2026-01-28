# image-similarity

This is a small script I wrote over an evening for a game I ran with friends on our New Years Eve getaway. It calculates a feature vector using an EfficientNetB3 model pretrained on ImageNet and then calcualtes the cosine distance between feature vectors.

## Run

1. Install the requirements
2. Add your originals to the `./originals` folder
3. Add photos to compare against to some folder with the same names and run the script:

```sh
python3 main.py {submission directory}
```

If you wish, you can try it yourself with the images under `./examples`
