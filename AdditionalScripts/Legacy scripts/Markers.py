import pandas as pd

list_ = ['короновируса нет', 'пандемии нет', 'эпидемии нет', 'вакцина', 'причина', 'вируса нет',
         'говорят', 'коронавирус из', 'вирус из', 'разработали', 'создали', 'теория', 'заговор', 'слух', 'лаборатории',
         'Кениг', 'Кёниг', 'америка', 'сша', 'бананы', 'яблоки', 'гуляющих', '2015', 'убийство пациентов', 'гвардия',
         'выгодно', 'привитые', 'распространяют', 'войска', 'за шутки', 'Валдай', 'профилактика', 'алкоголь',
         'продлен', 'войска', 'чип', 'пенсионер', 'придумали', ' чат ', 'стягив', 'статья', 'Times', 'Таймс',
         'Rammstein', 'Линдеманн']
pd.set_option('display.max_colwidth', None)
path = '../FinalData/Test-17.csv'
PdFromFinalData = pd.read_csv(path, index_col='number',sep=',')
for i in range(len(PdFromFinalData)):
    if str(PdFromFinalData['space'][i]) == 'nan':
        if any(word in PdFromFinalData['text'][i] for word in list_):
            print('{', i, '|', len(PdFromFinalData), '|', PdFromFinalData['space'].count(), '} →',
                  PdFromFinalData['text'][i])

            marker = input('0 - обычный текст, 1 - слух, 2 - удалить, 4 - exit\n')
            if marker == str(4):
                break
            else:
                PdFromFinalData['space'][i] = int(marker)
                PdFromFinalData.to_csv(path)
    else:
        pass
