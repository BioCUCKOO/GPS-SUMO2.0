# -*- coding: utf-8 -*-
# @Time    : 2022/12/14  14:20
# @Author  : Gou Yujie
# @File    : predict_github.py

class Predict:
    def __init__(self, inputfile, kind, outdic, modelfo):
        self.input = inputfile
        self.kind = kind
        self.outdic = outdic
        self.modelfo = modelfo

    def checkfile(self):
        dicnames = ['asa_result', 'ZFilepathPSSM', 'ZFilepathresult', 'ZID']
        for name in dicnames:
            path = self.outdic + self.kind + '/' + name + '/'
            if not os.path.exists(path):
                os.mkdir(path)
            else:
                shutil.rmtree(path)
                os.mkdir(path)

    def protein(self):
        nameprot = {}
        file = open(self.input, 'r').readlines()
        protline=''
        name=''
        for line in file:
            flag = 0
            if line.startswith('>'):
                p = re.compile(r'[,$()#+&*]')
                name = line.strip().strip('>')
                name = re.sub(p, '-', name)
                protline = str()
                continue
            if flag == 0:
                protline += line.strip()
            nameprot[name] = protline
        for name,seq in nameprot.items():
            out=open(outdic+"%s.fas"%(name),"w")
            out.write('>'+name+'\n'+seq)
            out.close()
        return nameprot

    def readfasta_site(self, nameprotdic):
        prot_seqs_site = {}
        for name, protline in nameprotdic.items():
            if len(protline) < 61:
                protline = protline + '*' * (61 - len(protline))
            ksite = [i.start() for i in re.finditer('K', protline)]
            for site in ksite:
                if site < 30:
                    seq = '*' * (30 - site) + protline[:site + 31]
                elif site >= len(protline) - 30:
                    seq = protline[site - 30:] + '*' * (31 - (len(protline) - site))
                else:
                    seq = protline[site - 30:site + 31]
                prot_seqs_site[name + '~' + str(site + 1)] = seq.replace('U', '*')
        return prot_seqs_site

    def readfasta_SIM(self, nameprotdic):
        prot_seqs_SIM = {}
        for name, seq in nameprotdic.items():
            flag = 0
            idx_5 = 0
            for idx, i in enumerate(seq):
                if idx == idx_5:
                    flag = 0
                if flag == 1:
                    continue
                core = seq[idx:idx + 5]
                if core.startswith('I') or core.startswith('V') or core.startswith('L'):
                    ret = re.findall("[IVL]{1}", core)
                    if len(ret) >= 3:
                        flag = 1
                        idx_5 = idx + 5
                        if idx < 30:
                            overseq = '*' * (65 - len(seq[:idx + 35])) + seq[:idx + 35]
                        elif idx + 5 > len(seq) - 30:
                            overseq = seq[idx - 30:] + '*' * (65 - len(seq[idx - 30:]))
                        else:
                            overseq = seq[idx - 30:idx + 35]
                        prot_seqs_SIM[name + '~' + str(idx + 1) + '-' + str(idx + 5)] = overseq.replace('U', '*')

        return prot_seqs_SIM

    def wrongmark(self):
        return 'No ' + self.kind + ' sequence found!'

    def getfeat(self, nameprotdic, pepdic, cutoff, kindup):

        self.kind = kindup
        namelist = list(pepdic.keys())
        peplist = list(pepdic.values())
        transdata = seq2num(peplist)
        modelpath = self.modelfo + '/' + self.kind + '/'
        features = ['transformer']
        lisarr = []
        scores = np.zeros((len(peplist), 1))

        for loc, feat in enumerate(features):

            if feat == 'transformer':
                feat = 'transformer_feat'
                model = load_model(modelpath + feat + '.model')
                outfeat = model.predict(transdata, batch_size=10)

            else:
                model = load_model(modelpath + feat + '.model')
                outfeat = model.predict(np.array(lisarr[loc]))

            outfeat = outfeat[:, ~np.isnan(outfeat).any(axis=0)]
            outfeat = np.array(outfeat)[:, 1].reshape(-1, 1)
            scores = np.hstack((scores, outfeat))

        scores = np.around(scores[:, 1:],4)

        outframe = pd.DataFrame(index=namelist, columns=['index', 'ID', 'Position', 'Peptide', 'Score','Cut-off', 'Type','Peptide_k','Peptide_sim','link'])
        outframe['index'] = namelist
        outframe['ID'] = outframe['index'].apply(lambda x: x.split('~')[0])
        outframe['Position'] = outframe['index'].apply(lambda x: x.split('~')[1])
        outframe['Peptide_k'] = outframe['index'].apply(lambda x: pepdic[x][23: -27])
        outframe['Peptide_sim'] = outframe['index'].apply(lambda x: pepdic[x][23: -23])
        outframe['Peptide'] = outframe['index'].apply(lambda x: pepdic[x])
        outframe['Score'] = scores
        outframe['Cut-off'] = [cutoff for i in range(len(namelist))]
        outframe['Type'] = [self.kind for i in range(len(namelist))]
        outframe['link'] = outframe['index'].apply(lambda x: sitedic[x.split('~')[0]] if x.split('~')[0] in sitedic.keys() else 'NA')
        outframe['uniprot'] = outframe['index'].apply(lambda x: uniprotdic[x.split('~')[0]] if x.split('~')[0] in uniprotdic.keys() else 'NA')
        aaindex={}
        asadic={'A':93.7,'L':173.7,'R':250.4,'K':215.2,'N':146.3,'M':197.6,'D':142.6,'F':228.6,'C':135.2,'P':0,'Q':77.7,'S':109.5,'E':182.9,'T':142.1,'G':52.6,'W':271.6,'H':188.1,'Y':239.9,'I':182.2,'V':157.2,'U':0,'X':0,'*':0}
        na = list(asadic.values())
        asadic = {k: (v - min(na)) / (max(na) - min(na)) for k, v in asadic.items()}

        for prot in nameprotdic.keys():
            aaindex[prot] = ''
            seq = nameprotdic[prot]
            for i in seq:
                aaindex[prot] += i + ';' + str(asadic[i]) + '~'
        outframe['aaindex'] = outframe['ID'].apply(lambda x: aaindex[x])
        def label(outframe):
            if outframe['Type'] == 'SUMO-interaction':
                return outframe['Peptide_sim']
            elif outframe['Type'] == 'Sumoylation':
                return outframe['Peptide_k']

        outframe['Peptide'] = outframe.apply(label, axis=1)
        outframe = outframe.sort_values(by=['ID', 'Position'], ascending=True)
        wholeframe = outframe[['ID', 'Position', 'Peptide', 'Score', 'Cut-off', 'Type', 'link','aaindex', 'uniprot']]
        for ind in wholeframe.index:
            cplmid = wholeframe.loc[ind, 'link']
            if cplmid == '':
                wholeframe.loc[ind, 'Source'] == 'Pred.'
            else:
                prot = wholeframe.loc[ind, 'ID']
                try:
                    if int(wholeframe.loc[ind, 'Position']) in needdic[prot]:
                        wholeframe.loc[ind, 'Source'] = 'Exp.'
                    else:
                        wholeframe.loc[ind, 'Source'] = 'Pred.'
                except:
                    wholeframe.loc[ind, 'Source'] = 'Pred.'
            uniprotid = wholeframe.loc[ind, 'uniprot']
            if uniprotid in set(ppi[0]):
                protname = list(ppi[ppi[0] == uniprotid][1])[0]
                ppiprot = list(ppi[ppi[0] == uniprotid][3])[0]
                se = "<a href =https://www.uniprot.org/uniprotkb/%s/entry target='_blank'>%s</a>" % (protname, ppiprot)
                wholeframe.loc[ind, 'ppi'] = se
            else:
                wholeframe.loc[ind, 'ppi'] = 'NA'

        wholeframe = wholeframe[['ID', 'Position', 'Peptide', 'Score', 'Cut-off', 'Type', 'link', 'Source','aaindex', 'ppi']]
        getline = []
        for i in wholeframe.index:
            if float(wholeframe.loc[i, 'Score']) > float(wholeframe.loc[i, 'Cut-off']):
                getline.append(i)
        wholeframe = wholeframe.loc[getline, :]
        for i in wholeframe.index:
            if not re.findall('-', i.split('~')[1]):
                temp = wholeframe[(wholeframe['ID'] == i.split('~')[0]) & (wholeframe['Type'] == "Sumoylation")]
                totsim = [j for j in wholeframe.loc[i, 'aaindex'].split('~') if j.startswith("K")]
                countdic[i.split('~')[0] + '~site'] = str(len(temp)) + ',' + str(len(totsim) - len(temp))
            else:
                temp = wholeframe[(wholeframe['ID'] == i.split('~')[0]) & (wholeframe['Type'] == "SUMO-interaction")]
                totsim = [j for j in pepdic.keys() if re.findall('-', j) and j.split('~')[0] == i.split('~')[0]]
                countdic[i.split('~')[0] + '~sim'] = str(len(temp)) + ',' + str(len(totsim) - len(temp))
        return wholeframe,outframe

from threading import Thread
class getThread(Thread):
    def __init__(self,func,args):
        Thread.__init__(self)
        self.args = args
        self.func = func
        self.result = None
    def predict(self):
        self.result=self.func(*self.args)
        return self.result
def func(type,dic):
    global result
    if type=='Sumoylation':
        sitedic = redfas.readfasta_site(dic)
        if sitedic:
            result1,outframe = redfas.getfeat(dic, sitedic,sumoylation_cutoffdic[sumotion],type)

        else:
            result1 = pd.DataFrame()
            outframe = pd.DataFrame()
        plusdic = {}
        for k, v in countdic.items():
            if k.split('~')[0] + '~site' not in countdic.keys():
                plusdic[k.split('~')[0] + '~site'] = '0;' + str(len(outframe))
        countdic.update(plusdic)
        return result1
    elif type=='SUMO-interaction':
        sitedic = redfas.readfasta_SIM(dic)
        if sitedic:
            result2,outframe = redfas.getfeat(dic,sitedic, SIM_cutoffdic[sim],type)
        else:
            result2 = pd.DataFrame()
            outframe = pd.DataFrame()
        plusdic = {}
        for k, v in countdic.items():
            if k.split('~')[0] + '~sim' not in countdic.keys():
                plusdic[k.split('~')[0] + '~sim'] = '0;' + str(len(outframe))
        countdic.update(plusdic)
        return result2
import threading
def main():
    global countdic
    countdic = {}

    result = pd.DataFrame()
    for i in kinddic[begkind]:
        resulttemp = func(i, protdic)
        result = pd.concat([result, resulttemp])

    if begkind =='Sumoylation':
        result['count'] = result['ID'].apply(lambda x: countdic[x + '~site'] + '*,;')
    elif begkind == 'SUMO-interaction':
        result['count'] = result['ID'].apply(lambda x: ',;*'+countdic[x + '~sim'])
    else:
        result['count'] = result['ID'].apply(lambda x: countdic[x + '~site'] + '*'+countdic[x + '~sim'])
    return result

import numpy as np
import re
import pandas as pd
import os
import pickle
import sys
import shutil
import scipy.stats as st
from pandas import DataFrame
from tensorflow.keras.models import load_model
from threading import Thread
from copy import deepcopy

sumotion=sys.argv[1]
sim=sys.argv[2]
begkind=sys.argv[3]
inputfile=sys.argv[4]
outdic=sys.argv[5]
species=sys.argv[6]
modelfo="/data/www/sumo/predict/models/"
sumoylation_cutoffdic={'All':0,'High':0.85,'Medium':0.55,'Low':0.3}
SIM_cutoffdic = {'All': 0, 'High': 0.50, 'Medium': 0.45, 'Low': 0.40}
kinddic = {'Both': ['Sumoylation', 'SUMO-interaction'], 'Sumoylation': ['Sumoylation'],
           'SUMO-interaction': ['SUMO-interaction']}
wholeframe = pd.DataFrame()
redfas = Predict(inputfile, begkind, outdic, modelfo)
protdic = redfas.protein()
cplm = pd.read_csv("/data/www/sumo/predict/Sumoylation.txt", sep='\t', index_col=0, header=None)
sitedic = {}
needdic={}
uniprotdic={}
ppi=pd.read_csv("/data/www/sumo/predict/allppi_species.txt",sep='\t',header=None)
if species!='all':
    ppi = ppi[ppi[2] == species]
    uniprotframe = pd.read_csv("/data/www/sumo/webcomp/%s.txt" % (species), sep='\t', header=None)
else:
    uniprotframe=pd.DataFrame(columns=[1,2,3])
for k, v in protdic.items():
    if k in sitedic.keys() and k in needdic.keys():
        break
    if v in set(cplm[6]):
        sitedic[k] = [i for i in cplm[cplm[6] == v].index][0]
        needdic[k] = [i for i in cplm[(cplm[6] == v) & (cplm[3] == 'Sumoylation')][2].tolist()]
        uniprotdic[k] = [i for i in cplm[cplm[6] == v][1]][0]
        continue
    if v in set(uniprotframe[3]):
        uniprotdic[k] = [i for i in uniprotframe[uniprotframe[3] == v][0]][0]
wholeframe=main()
wholeframe=wholeframe[['ID', 'Position', 'Peptide', 'Score','Cut-off', 'Type','link','Source','ppi','aaindex','count']]
wholeframe = wholeframe.sort_values(by=['ID','Score'],ascending=False)
wholeframe.to_csv(outdic+ ".txt", sep='\t',index=None)





