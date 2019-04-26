import os as OS
from codecs import encode, decode
from bson.code import Code

vocab={}
def getallfiles(Path):
    fileList=list()
    dirList=OS.listdir(Path)
    for deep1path in dirList:
        deep2path=OS.path.join(Path,deep1path)
        if OS.path.isdir(deep2path):
            getallfiles(deep2path)
        else:
            fileList.append(deep2path)
    return fileList



def build_vocab(files):
    Nc1=0
    Nc2=0
    total_terms=0
    for text in files:
        doc=list()

        C1=0
        C2=0
        if text.__contains__('fortnow'):
            C1=1
            Nc1+=1
        else:
            C2=1
            Nc2+=1
        lines=open(text).readlines()
        stripChars='<>?/)!=([]{};:-"\'.%,#@* \n\t'
        for line in lines:
            tokens=line.strip(stripChars).split()
            for token in tokens:
                word = decode (token.strip(stripChars) , 'latin2' ,'ignore' )
                if len(word)>=4 and word.isalpha():
                    total_terms+=total_terms
                    if vocab.__contains__(word):
                        itemm=vocab[word]
                        if C1==1:
                            itemm["C1"]+=C1
                            itemm["C2"]+=C2
                            vocab[word]=itemm
                    else:
                        item={}
                        item.__setitem__("C1",C1)
                        item.__setitem__("C2",C2)
                        vocab.__setitem__(word,item)
    return vocab, Nc1, Nc2, total_terms


def compute_parameters(vocab):
    pass

def fit_model(vocab,Nc1,Nc2):
    N=Nc1+Nc2
    Pc1=N/float(Nc1)
    Pc2=N/float(Nc2)

    for words in vocab:
        pass







if __name__ == '__main__':
    V, Nc1, Nc2, NT=build_vocab(getallfiles("TestSet"))
    print len(V), Nc1, Nc2,NT