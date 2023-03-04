# Image-caption-generator

This model is trained on [Flickr8k](https://www.kaggle.com/datasets/nunenuh/flickr8k) dataset to generate captions given an image.

It achieves the following results on the evaluation set:
- eval_loss: 0.2536
- eval_runtime: 25.369
- eval_samples_per_second: 63.818
- eval_steps_per_second: 8.002
- epoch: 4.0
- step: 3236

# Running the model using transformers library

1. Load the pre-trained model from the model hub
    ```python
    from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
    import torch
    from PIL import Image
    
    model_name = "bipin/image-caption-generator"

    # load model
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ```

2. Load the image for which the caption is to be generated
    ```python
    img_name = "flickr_data.jpg"
    img = Image.open(img_name)
    if img.mode != 'RGB':
        img = img.convert(mode="RGB")
    ```

3. Pre-process the image
    ```python
    pixel_values = feature_extractor(images=[img], return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    ```

4. Generate the caption
     ```python
      max_length = 128
      num_beams = 4

      # get model prediction
      output_ids = model.generate(pixel_values, num_beams=num_beams, max_length=max_length)

      # decode the generated prediction
      preds = tokenizer.decode(output_ids[0], skip_special_tokens=True)
      print(preds)
     ```

## Training procedure
The procedure used to train this model can be found [here](https://bipinkrishnan.github.io/ml-recipe-book/image_captioning.html).

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 5

### Framework versions

- Transformers 4.16.2
- Pytorch 1.9.1
- Datasets 1.18.4
- Tokenizers 0.11.6
