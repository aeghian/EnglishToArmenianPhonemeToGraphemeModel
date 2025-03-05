import pandas as pd
import re
import itertools
import multiprocessing
traditional_armenian_wordlist = pd.read_csv('text_list.csv')
armenian_english_key = {
    'ա':['ah','a'], 
    'բ':['p'], 
    'գ':['c','k'], 
    'դ':['t'], 
    'ե':['e', 'eh'], 
    'զ':['z'],
    'է':['e','eh'], 
    'ը':['u','uh'], 
    'թ':['t'], 
    'ժ':['zh','zsh'],
    'ի':['i','ee'],
    'լ':['l'], 
    'խ':['kh'], 
    'ծ':['dz','ds'], 
    'կ':['g'], 
    'հ':['h'], 
    'ձ':['ts','tz'], 
    'ղ':['gh'], 
    'ճ':['j'], 
    'մ':['m'], 
    'յ':['y'], 
    'ն':['n'], 
    'շ':['sh'], 
    'ո':['o','oh'], 
    'չ':['ch'], 
    'պ':['b'],
    'ջ':['ch'], 
    'ռ':['r'], 
    'ս':['s'], 
    'վ':['v'], 
    'տ':['d'], 
    'ր':['r'], 
    'ց':['ts','tz'], 
    'ւ':['v'], 
    'փ':['p'],
    'ք':['c','k'],
    'օ':['o','oh'],
    'ֆ':['f'],
    'ու':['oo', 'u', 'ooh'], 
    'իւ':['yu','you','yoo'],
    'ոյ':['ooy','ouy', 'uy'], #may need to ammend this list
    'ուո':['vo','voh'], 
    'ուօ':['vo','voh'],
    'ուի':['vi', 'vee'], 
    'ուա':['va','vah'], 
    'ուէ':['ve','veh'], 
    'ուե':['ve','veh'], 
    'ուը':['vu','vuh']
}

def InitialClean(row):
    cleaned_word = []
    if type(row['0']) == str:
        for letter in row['0']:
            if re.search('[ա-ֆ]', letter) == None:
                letter = 'եւ'
            cleaned_word.append(letter)
    return ''.join(cleaned_word)

def CreateEnglishPossibilities(row, armenian_english_key):
    if type(row['0']) == str:
        grapheme_list = BreakWordToLargestGraphemes(row['cleaned_words'], armenian_english_key)
        english_character_possibilities = []
        initial_english_possibilities = []
        for grapheme in grapheme_list:
            english_character_possibilities.append(armenian_english_key[grapheme])
        english_combo_list = list(itertools.product(*english_character_possibilities))
        for english_combo in english_combo_list:
            initial_english_possibilities.append(''.join(english_combo))
        final_english_possibilities = AddFirstLetterExceptions(row['cleaned_words'], initial_english_possibilities)
        return final_english_possibilities

def BreakWordToLargestGraphemes(armenian_word, armenian_english_key):
    multiletter_graphemes = []
    for grapheme in armenian_english_key.keys():
        if len(grapheme) > 1:
            multiletter_graphemes.append(grapheme)
    multiletter_graphemes.sort(key=len)
    multiletter_graphemes.reverse()
    regex_expression = CreateRegexExpression(multiletter_graphemes)
    raw_grapheme_list = re.split(regex_expression, armenian_word)
    cleaned_grapheme_list = CleanGraphemeList(raw_grapheme_list, multiletter_graphemes)
    return cleaned_grapheme_list

def CreateRegexExpression(multiletter_graphemes):
    regex_expression = '('
    for multiletter_grapheme in multiletter_graphemes:
        regex_expression += (multiletter_grapheme+'|') 
    regex_expression = regex_expression[:-1] + ')'
    return regex_expression

def CleanGraphemeList(raw_grapheme_list, multiletter_graphemes):
    cleaned_grapheme_list = []
    for grapheme in raw_grapheme_list:
        if grapheme in multiletter_graphemes:
            cleaned_grapheme_list.append(grapheme)
        elif len(grapheme) > 0:
            for letter in grapheme:
                cleaned_grapheme_list.append(letter)
    return cleaned_grapheme_list

def AddFirstLetterExceptions(cleaned_word, initial_english_possibilities):
    first_character = {'յ':'h','ո':'v','ե':'y'}
    final_english_possibilities = []
    if cleaned_word[0] not in first_character.keys():
        return initial_english_possibilities
    return [first_character[cleaned_word[0]] + initial_english_possibilities for initial_english_possibilities in initial_english_possibilities]

def CreateTranslitToArmenian(traditional_armenian_wordlist, armenian_english_key, batch_id):
    translit_to_armenian = pd.DataFrame()
    for index, row in traditional_armenian_wordlist.iterrows():
        true_index = index - traditional_armenian_wordlist.first_valid_index()
        if true_index%160 == 0: 
            print(f"{batch_id}{true_index/1600}")
        armenian_word = row['cleaned_words']
        armenian_graphemes = BreakWordToLargestGraphemes(armenian_word, armenian_english_key)
        for english_possibility in row['english_possibilities']: 
            translit_to_armenian = translit_to_armenian.append(pd.DataFrame({'english_possibility':[english_possibility],'armenian_graphemes':[armenian_graphemes]}),ignore_index=True) 
    translit_to_armenian.to_csv(f'{batch_id}text_list.csv', index=False)

traditional_armenian_wordlist['cleaned_words'] = traditional_armenian_wordlist.apply(lambda row: InitialClean(row), axis=1)
print('first')
traditional_armenian_wordlist['english_possibilities'] = traditional_armenian_wordlist.apply(lambda row: CreateEnglishPossibilities(row,armenian_english_key), axis=1)
print('second')
p = multiprocessing.Process(target=CreateTranslitToArmenian, args=(traditional_armenian_wordlist[2:50000], armenian_english_key, 'ID1-'))
p.start()
p = multiprocessing.Process(target=CreateTranslitToArmenian, args=(traditional_armenian_wordlist[50000:100000], armenian_english_key, 'ID2-'))
p.start()
p = multiprocessing.Process(target=CreateTranslitToArmenian, args=(traditional_armenian_wordlist[100000:150000], armenian_english_key, 'ID3-'))
p.start()