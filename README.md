# Agent for Newsletter Creation
This repository contains a NLP project focused on an AI agent for automating the process of newsletter creation. 
The project leverages the LangChain, HuggingFace, Torch frameworks etc to create the bot. 

## Technologies
Modules used:
1. Langchain (for Document Loaders, Text Splitters, Chain creation)
2. Transformers (For summarization models)
3. Bitsandbytes (For quantization)
4. Diffusers (For image creation)

The project utilizes the "microsoft/Phi-3-mini-128k-instruct" model for summarizing.
You can access it directly by using below code:
```
model_id = "microsoft/Phi-3-mini-128k-instruct"
```
and
the "stabilityai/stable-diffusion-xl-base-1.0" model for image creating.
You can access it directly by using below code:
```
pipe = StableDiffusionXLPipeline.from_pretrained(
  "stabilityai/stable-diffusion-xl-base-1.0",
  torch_dtype = torch.float16,
  variant = "fp16",
  use_safetensors = True
)
```

## Project Overview
The core of the project is to automate the process of newsletter creation.
The python file is included in this repository provides a detailed walkthrough of the model implementation.

## Procedure to building the model
### Data processing
Get the url/website link as input, use dataloaders to load as Document type, save the contents in content.txt

### Summarizing and title creation
Use the content.txt and the summarization model by transformers library to create summary and save the text to summary.txt, use the same model and the summary for creating a catchy title

### Image creation
Use the summarization model to create a image_prompt and use that and title as input for diffusion model, save the image to newsletter_image.png

### Assembling
To bring together all the collected data and assemble and output

## Note: 
The url is to be inputted by the user in article_url for seamless working of the code.

## Future ideas:
To use Gradio interface for seamless interaction, better diffusion and summarization models for quality output, and better html templates.
