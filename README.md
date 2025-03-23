https://github.com/user-attachments/assets/eb957e90-f191-4bf4-a3e3-2e82e996a2a8

**INTRODUCTION**

The goal of this algorithm is to create a solution allowing people able to speak Armenian to write in the language without knowing the alphabet or learning a specially mapped Armenian keyboard configuration. The current model is designed for Wetsern Armenian pronunciations. This is done by converting Armenian words spelled in English phonetically (phonemes) to their appropriate Armenian counterpart (graphemes). This model needed to be deployed in a browser extention and run locally. This choice was made because text data is potentially sensitive, and users may not want this data exposed to the cloud. Below sections describing the algorithm choice, data gathering process, training, conversion into Tensorflow, implementation into the brwoser addon, and included files have been provided.

**CHOOSING ALGORITHM**

The initial goal of this project was not to use machine learning, but brute force algorithms created exponentially scaling options. The first iteration of this algorithm manually created all possible combinations, but for each phoneme with multiple grapheme options the possibilities stacked. For example, the phoneme ch could indicate pronunciation like charlie, chiropractor, or even a separate c and h sound. This could result in possible Armenian graphemes (pronounced in Western Armenian) as 'գ,' 'ք,' 'ջ,' or 'չ.' If a word uses just two of these phonemes it would result in 16 possible combinations to check (4 options times 4 options). It may be possible to prune the possibilities based on the possible Armenian words, but this may have taken more time than the process of training a machine learning model.

After deciding on using machine learning to predict the possible Armenian words, research was done to see how others have used machine learning to solve similar problems. Lots of credit to fehiepsi for the following article https://fehiepsi.github.io/blog/grapheme-to-phoneme/. This project provided a blueprint to make the initial model which was extremely helpful to get something up and running very quickly. The initial pytorch model is very closely tied to fehiepsi's project. This article used a model that replicated the following research paper https://arxiv.org/abs/1506.00196.

**DATA GATHERING**

Collecting the appropriate data for this model was tricky because there was no structured data source to work with. The initial strategy was to use online Armenian dictionaries to compare the phonetic system used to the actual word entry. This process created two challenges. First, gathering this data was challenging because online Armenian dictionaries were unwilling to share their data directly or provide an API. Scraping the data also presented challenges due to active anti-scraping measures employed by some Armenian dictionaries. The second challenge was the extra model architecture and training work required to make this process work. One model would be needed to convert English letters to the phonetic system used by dictionaries, and a second model would convert this phonetic system to Armenian. Besides additional development complexity two models would require more computing resources which is not ideal for the goal of local inference. 

The next strategy was collecting examples of Armenian words spelled in English letters and using the model architecture referenced above to do a direct conversion. While this strategy was more successful, scraping transliterations from books was not an ideal source of training data for the model. Since more formal sources have provided Armenian literature in English transliterations, there was no variation to the phonemes used to represent Armenian words. This does not make for a good data source because the model will only learn static patterns. Effectively, this solution would be worse than a deterministic solution since it would have the same rigidity while likely being slower and larger.

Finally, this strategy was evolved by generating a data set. This data set was created by capturing a large volume of Armenian words and creating an algorithm to generate all possible ways to represent the words in English letter phonemes based on Western pronunciation. By feeding this wide range of examples into the architecture detailed above a very effective model was produced. Although this model does not always produce the correct word on the first try every time, it is very accurate within its top 10 guesses. This process has some drawbacks since it does not use naturally created phonetics as an example, but the initial process used to generate the training data could be modified to include any important rule change. 

**TRAINING MODEL**

Once the above architecture and data sources were established a Google Colab notebook was used for training. Overall, the model accuracy proved very functional considering the model's top result was right 87% of the time and the model contained the right answer within the top 10 resposes 98% of the time. These figures were based on a small random sample of 100 words, but they help show the directional effectiveness of a relatively simple model design process. The model architecture was only shifted to accomdate conversion to Tensorflowjs, and the model's data was generated by estimating how people will type various letters. The input data could easily be modified to accomdate changes to the model output.   

**CONVERTING MODEL**

The model was initially trained in pytorch with the understanding that ONNX could be used to convert the model to a format suitable for a browser extension. Unfortunately, ONNX could not reliably convert the model to a useful state. The next method tried was to deploy the model in c++. Although this was successful using Pytorch's built in API, browser extensions depricated the ability to run c++ in a fashion needed for the current addon. Finally, the model was converted to Tensorflow due to Tensorflowjs' ability to run natively within the browser. Due to the very helpful nobuco program developed by Alexander Lutsenko (https://github.com/AlexanderLutsenko/nobuco) this process was much less painful than it otherwise would have been. The program was a little difficult to understand at first, but working through the errors provided by the program served as a step by step process for conversion as opposed to remaking the model from scratch in Tensorflow. Since this project succesfully utilized this tool, others should feel free reach out to me on Linkedin (https://www.linkedin.com/in/armen-eghian-6979b01a1/) if they have any questions about deploying this tool in other contexts.

**IMPLEMENTING MODEL**

Once the pytorch model was converted to Tensorflow, implementation was a very easy process. Tensorflowjs is called by the javascript addon and the proper model is loaded into the program. The addon is set up so new models can be placed in the file path and referenced by the configuration page as long as these are Tensorflow models exported in the SaveModel format with a 2d input of ["word length",1] and an output of ["ranked output options","max word length"]. Due to the setup of the current model, the input is reversed before being used. This should be kept in mind when implementing additional models or using this model in other contexts. For more information about the browser addon the project can be found here https://github.com/aeghian/TfjsTransliteration.

**FILE EXPLANATIONS**

Below is a description of each folder or individual file:
    
- tensorflowjs_model_32_max: This folder contains the Tensorflowjs model created to transofrm English phonemes to Armenian graphemes. This model is designed for 32 character inputs.
- cmudict.dict (1 & 2): This file contains the training data used to create the current model. It has been split into two parts due to file size restrictions in github.    
- g2p_nobuco.py: This file contains the pytorch model architecture and nobuco modifications needed to convert the model to Tensorflowjs.    
- manipulation.ipynb: This file was used to convert the .csv training data into the proper .dict format so it could be used for training.    
- pytorch_3_13_2023.pth: This is the original pytorch model trained on 3/13/2023.    
- text_cleaner_script.py: This file was used to generate a .csv file of training data from text_list.csv.    
- text_list.csv: This file contains a list of Armenian words. 
