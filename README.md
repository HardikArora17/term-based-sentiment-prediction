# term-based-sentiment-prediction

For properly running the code follow the below steps:

1) cd path/absa
2) Install all the dependencies using the requirement.txt
3) If you want to train the model its code is availabel at notebooks/main_intern_train.py
4) For getting the results for any csv file you can run the code at src/evaluation.py.

   Results are stored at data/results/test_data.csv

5) If you want to see a live demo of the project you can run it by typing the following command: streamlit run src/evaluation_streamlit.py



Other things about the model can be found below

**MODEL DESCRIPTION**

●	**Word embeddings layer** - For calculating embeddings for the sentence as well as for the aspect phrase I tried different ways :

1)	Transformer based embeddings(Bert and Roberta) 
2)	Normal nn.Embeddings layer available in pytorch without any externa weights
3)	Static Glove Embeddings  **(finally used in model)**


The problem with approach 1 is that it gives you an  embedding with 768 features ,which is practically possible for the model to learn due to less training data

The problem with approach 2 was better than approach 1 but was not giving good results maybe again due to lack of learning 

Due to all this I used approach 3 for creating the word embeddings , I used standard           6B.100d glove version for creating 100 dimension ,sentence and word embeddings , I also tried with 50 dimension and 200 dimension but this was giving the best results in our favour.

●	**BILSTM LAYER**-  Since I used static word embeddings  , words are actually deprived of any context in their vicinity. To solve this I used  a simple bi-lstm network for  generating contextualized word embeddings

●	**Feed Forward layers**-  After getting embeddings for both the aspect phrase as well as for the text , I used some linear layers so that the model could find useful features and establish relationships between input and output. 

●	**Softmax Layer**-  At last I used a softmax layer for predicting the sentiment. 


Since this is a multi class problem I used a cross entropy loss as my loss function . I used Adam optimizer with a learning rate of 0.001 . All the dropout layers have a dropout rate of 0.3. I trained the model for 30 epochs ,with a batch size of 16 on Google Colab.




For the purpose of evaluating my model I used  f1 score (both label wise and overall)

Here are the scores for my model on the validation set:

