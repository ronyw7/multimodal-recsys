# cs224w-proj

## Dataset Curation
### Google Review Dataset
To download the Google Review dataset, available as `.json` files, run the following commands:
```
# This downloads the filtered subset ~112M
wget https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/googlelocal_restaurants/filter_all_t.json 

# This downloads the entire dataset ~1.0G
wget https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/googlelocal_restaurants/image_review_all.json
```

### Retrieving Image Files
For each linked user review image, we need to access Google's user content endpoint to retrieve it. We store the retrieved images in `./data`.

Since the release of the Google Reviews dataset, some user uploaded content have been deleted or removed. We report the number of valid user review images we were able to collect for each of the train, val, test splits.
- `./data/train` contains 159,426 images
- `./data/val` contains 20,035 images
- `./data/test` contains 20,073 images
