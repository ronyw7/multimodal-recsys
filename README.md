# cs224w-proj

## Dataset Curation
### Google Restaurants Dataset
To download the Google Restaurants dataset, available as `.json` files, run the following commands:
```
# This downloads the filtered subset ~112M
wget https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/googlelocal_restaurants/filter_all_t.json 

# This downloads the entire dataset ~1.0G
wget https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/googlelocal_restaurants/image_review_all.json
```

### Retrieving Image Files from Google
For each linked user review image, we need to access Google's user content endpoint to retrieve it. We store the retrieved images in `./data`.

Since the release of the Google Restaurants dataset, some user-uploaded contents have been deleted or removed. We report the number of valid user review images we were able to collect for each of the train, val, test splits.
- `./data/train` contains 159,426 images
- `./data/val` contains 20,035 images
- `./data/test` contains 20,073 images

Additionally, we have 87,013 text reviews in the training set, 10,860 in the validation set, and 11,015 in the test set.

### Generating Image and Text Embeddings
Image embeddings are generated using the `model.encode_image()` API. Results are saved to `embeddings_pics_[train|val|test].npz`

To load it, 
```
embeddings = np.load('embeddings_pics_train.pkl', allow_pickle=True)
# For model training, use `torch.tensor()` to convert the numpy array into a torch tensor
key = 'AF1QipMn4wPFuEhb31cx8AzxY86qj6TH4uV8e3o_GARh'
embeddings = torch.tensor(embeddings[key])
```

Because the CLIP text encoder supports a max context window size of 77 tokens, we break down each text reviews into smaller chunks and generate embeddings for each chunk. For training purposes, these embeddings may be aggregated with `sum`, `mean`, etc. Results are saved to `embeddings_text_[train|val|test].npz`
