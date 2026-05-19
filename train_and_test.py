import json
import random
from collections import Counter
import torch
import torch.nn as nn
import nltk


#Loading and assigning logic

with open('alice_corpus.json', 'r') as file:
    corpus_words = json.load(file)

word_counts = Counter(corpus_words)
vocab_size = 3000
top_words = word_counts.most_common(vocab_size - 2)
word2idx = {"<PAD>": 0, "<UNK>": 1} #english to math (for inputs)
idx2word = {0: "<PAD>", 1: "<UNK>"} #math to english (for outputs)

for index, (word, count) in enumerate(top_words, start=2):
    word2idx[word] = index
    idx2word[index] = word

def convert_to_ids(word_list): #From words to numbers
    id_list = []
    for word in word_list:
        word_id = word2idx.get(word, word2idx["<UNK>"])
        id_list.append(word_id)
    return id_list

def convert_to_words(id_list): #From numbers to words
    return [idx2word.get(word_id) for word_id in id_list]



#Break words logic
def get_random_number(min_val, max_val):
    return random.randint(min_val, max_val)

def delete_letter(word):
    if len(word) <= 1: return word

    i = get_random_number(0, len(word) - 1)
    return word[:i] + word[i+1:]

def add_letter(word):
    i = get_random_number(0, len(word))
    alph = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
    j = get_random_number(0, len(alph) - 1)
    return word[:i] + alph[j] + word[i:]

def substitute_letter(word):
    if len(word) == 0: return word
    i = get_random_number(0, len(word) - 1)
    alph = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
    j = get_random_number(0, len(alph) - 1)
    return word[:i] + alph[j] + word[i+1:]

def mix_letters(word):
    if len(word)<2: return word
    i=get_random_number(0,len(word)-2)
    return word[:i]+word[i+1]+word[i]+word[i+2:]

def break_word(word):
    way = get_random_number(1, 4)

    if way == 1:
        return delete_letter(word)

    elif way == 2:
        return add_letter(word)

    elif way == 3:
        return substitute_letter(word)
    elif way==4:
        return mix_letters(word)

def target_words(dex):
    first = get_random_number(0, len(dex) - 5)

    words = []
    for i in range(first, first + 3):
        words.append(dex[i])

    clean_target = dex[first + 3]
    broken_target = break_word(clean_target)
    words.append(broken_target)

    return words, clean_target




#Pytorch

class Dictionary(nn.Module):
    def __init__(self,num_emb,emb_dim,hid):
        super(Dictionary, self).__init__()

        self.embedding=nn.Embedding(num_embeddings=num_emb,embedding_dim=emb_dim) #lookup table

        self.rnn=nn.LSTM(input_size=emb_dim,hidden_size=hid,batch_first=True) #update internal memory

        self.m=nn.Linear(in_features=hid,out_features=num_emb) #guess misspelled word


    def forward(self,input_x):
        input=torch.LongTensor(input_x) #first step
        embedded=self.embedding(input)

        lstm_out, hidden_states=self.rnn(embedded) # second step

        sliced = lstm_out[:, -1, :]# third step

        output=self.m(sliced) #forth step

        return output



# Training loop
def training_loop(X,Y):
    model=Dictionary(3000,64,128)
    loss_function = nn.CrossEntropyLoss() #for checking ai answer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #how much to tweak the answers

    for epoch in range(400):
        optimizer.zero_grad() #clear
        predictions = model(X)
        loss = loss_function(predictions, Y) #error between guess and correct answer
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item()}") #display

    return model



def final(trained_model, input_x, typo):
    # check the dictionary for misspelled words that look similar to the typo
    min_distance = 1000
    candidate_words = []
    
    for idx in range(2, vocab_size):
        word = idx2word.get(idx)
        if word is None: 
            continue
        dist = nltk.edit_distance(typo, word, transpositions=True)
        #closest match to the typo
        if dist < min_distance:
            min_distance = dist
            candidate_words = [word] 
        elif dist == min_distance:
            candidate_words.append(word)
            
            
    # make AI check only the list for a correct answer
    trained_model.eval()
    with torch.no_grad():
        predictions = trained_model(input_x) #AI generates scores for all 3000 words
        
        best_word = None
        best_score = -float('inf') # start with the lowest possible math score
        
        for word in candidate_words:
            word_id = word2idx[word]
            score = predictions[0][word_id].item() 
            if score > best_score:
                best_score = score
                best_word = word
                
        return best_word



x_train=[]
y_train=[]

print("generating dataset")
for _ in range (15000):
    words,clean= target_words(corpus_words)
    numeric_x=convert_to_ids(words)
    numeric_y=convert_to_ids([clean])[0]

    x_train.append(numeric_x)
    y_train.append(numeric_y)

y_tensor=torch.LongTensor(y_train)

print("training")

trained_model=training_loop(x_train,y_tensor)

print("tests:")

#memorising:

#test_sentence_numbers = [x_train[0]]
#test_answer_number = y_train[0]


#ai_guess = final(trained_model, test_sentence_numbers)
#correct_answer = convert_to_words([test_answer_number])[0]

# Print out the English words to see if it worked
#english_input = convert_to_words(x_train[0])
#print(f"Input Context + Typo: {english_input}")
#print(f"AI Guessed: {ai_guess}")
#print(f"Actual Answer: {correct_answer}")

#learning:
correct=0
for _ in range (100):

    test_words,clean_answer=target_words(corpus_words)
    numeric_x=convert_to_ids(test_words)
    typo=test_words[-1]
    ai_guess=final(trained_model, [numeric_x],typo)

    print(f"input and typo: {test_words}")
    print(f"ai guessed: {ai_guess}")
    print(f"correct answer: {clean_answer}")

    if (ai_guess==clean_answer):
        correct+=1
        print("correct!")
    else:
        print("incorrect")

print(f"final score: {correct}/100")
