# -*- coding: utf-8 -*-
# @Time    : 2022/12/14  14:25
# @Author  : Gou Yujie
# @File    : predict_total_github.py
# -*- coding: utf-8 -*-
# @Time    : 2022/11/17  19:53
# @Author  : Gou Yujie
# @File    : predict_total.py
import warnings
warnings.filterwarnings("ignore")
import threading,sys
from pandas import Series
from keras.utils import np_utils
import re
import pickle
import os
def readPeptide(pepfile,lr):
    data = []
    lr=30-lr
    with open(pepfile, 'r') as f:
        for line in f:
            if lr==0:
                data.append(line.rstrip().split('\t')[0])
            else:
                data.append(line.rstrip().split('\t')[0][lr:-lr])
    return data

def find_all_index(arr,item):
    return [i for i,a in enumerate(arr) if a==item]

def command_pssm(input_file, output_file, pssm_file, DBname):
    import subprocess
    cmd1=r'/data/www/icpc/iCTCF/ncbi-blast-2.11.0+/bin/psiblast -query ' + input_file + ' -db ' + DBname + ' -num_iterations 3 -num_threads 6 -out ' + output_file + ' -out_ascii_pssm ' + pssm_file
    subprocess.check_call(cmd1,shell=True)

def GetPSSM(ProSeq,OutDir,PSSMDir,DBname):

    records = list(SeqIO.parse(ProSeq, "fasta"))
    global aa1
    aa1 = {}
    for i, item in enumerate(records):
        if not os.path.exists(r'ZID/'):
            os.makedirs(r'ZID/')
        if  records[i].seq in ppf:
            seq1=records[i].seq
            if ppf[seq1].split('\t')[1]=='nan':continue
            input1=ppf[seq1].split('\t')[1]
            input2=ppf[seq1].split('\t')[2]
            if (not os.path.exists(input1)) or (not os.path.exists(input2)):continue
            out1 = PSSMDir + r'/' + records[i].id.split('|')[1] + '.pssm'
            out2 = PSSMDir[:-14]+r'/asa_result/' + records[i].id.split('|')[1] + '.txt'
            import shutil
            shutil.copy(input1,out1)
            shutil.copy(input2,out2)
            aa1[str(seq1)]=input1
        else:
            input_file=r'ZID/' + records[i].id.split('|')[1] +'.fa'
            SeqIO.write(records[i],input_file,'fasta')

            output_file = OutDir + r'/' + records[i].id.split('|')[1] + '.out'
            pssm_file = PSSMDir + r'/' + records[i].id.split('|')[1] + '.pssm'

            if not os.path.exists(pssm_file):
                command_pssm(input_file, output_file, pssm_file, DBname)
            aa1[str(records[i].seq)]=pssm_file

def StandardpPSSM(OldPSMMdir):
    listfile = os.listdir(OldPSMMdir)
    if listfile.__len__()==0:
        import shutil
        shutil.copy('../1.pssm',OldPSMMdir)
    listfile = os.listdir(OldPSMMdir)
    FinPSSM,num1=[],[]
    for i,eachfile in enumerate(listfile):
        num1.append(int(eachfile.split('.')[0]))
    num=sorted(num1)
    for i ,nn in enumerate(num):
        eachfile=str(nn)+'.pssm'
        with open(OldPSMMdir + '/' + eachfile,'r') as inputpssm:
            count = 0
            Dirdata=[]
            for line in inputpssm:
                count +=1
                if count <= 3:
                    continue
                if line.count('\n') == len(line):
                    break
                temp=line.strip().split()[2:22]
                Dirdata.append(temp)
            DirdataR=np.array(Dirdata)
            DirPSSM=np.reshape(DirdataR,(1,DirdataR.shape[0]*DirdataR.shape[1])).tolist()
            FinPSSM.append(DirPSSM[0])

    return FinPSSM,num
def psStandardpPSSM(OldPSMMdir):
    listfile = os.listdir(OldPSMMdir)
    if listfile.__len__()==0:
        import shutil
        shutil.copy('../1.txt',OldPSMMdir)
    listfile = os.listdir(OldPSMMdir)
    FinPSSM,FinPSSM0,FinPSSM1,num1=[],[],[],[]
    for i,eachfile in enumerate(listfile):
        num1.append(eachfile.split('.')[0])
    num=sorted(num1)
    namesout=[]
    for i,nn in enumerate(num):
        eachfile=str(nn)+'.txt'
        with open(OldPSMMdir + '/' + eachfile,'r') as inputpssm:
            count = 0
            Dirdata,Dirdata0,Dirdata1,Dirdata2,Dirdata3=[],[],[],[],[]
            Dirdata4, Dirdata5, Dirdata6, Dirdata7, Dirdata8 = [], [], [], [], []
            for line in inputpssm:
                count +=1
                if count <= 1:
                    continue
                if len(line)<2:
                    break
                temp=float(line.strip().split('\t')[3])
                Dirdata.append(temp)
                temp0=float(line.strip().split('\t')[8])
                Dirdata0.append(temp0)
                temp1=float(line.strip().split('\t')[9])
                Dirdata1.append(temp1)
                temp2=float(line.strip().split('\t')[10])
                Dirdata2.append(temp2)
                temp3 = float(line.strip().split('\t')[4])
                Dirdata4.append(temp3)
                temp4 = float(line.strip().split('\t')[5])
                Dirdata5.append(temp4)
                temp5 = float(line.strip().split('\t')[6])
                Dirdata6.append(temp5)
                temp6 = float(line.strip().split('\t')[7])
                Dirdata7.append(temp6)

            Dirdata3=Dirdata0+Dirdata1+Dirdata2
            Dirdata8=Dirdata4+Dirdata5+Dirdata6+Dirdata7
            FinPSSM.append(Dirdata)
            FinPSSM0.append(Dirdata3)
            FinPSSM1.append(Dirdata8)
            namesout.append(nn)
    return FinPSSM,FinPSSM0,FinPSSM1,num,namesout

def readweight(weight_file):
    weight = None
    with open(weight_file, 'r') as f:
        for i, line in enumerate(f):
            if i == 2 - 1:
                weight = np.array([float(x) for x in line.rstrip().split('\t')])
    return weight
def read_pssm(pssm_file):
    idx_res = (0, 4, 3, 6, 13, 7, 8, 9, 11, 10, 12, 2, 14, 5, 1, 15, 16, 19, 17, 18)
    aa = []
    pssm = []
    with open(pssm_file) as inputpssm:
        count = 0
        for line in inputpssm:
            count +=1
            if count <= 3:
                continue
            if line.count('\n') == len(line):
                break
            temp=line.strip().split()[2:22]
            aa_temp = line.strip().split()[1]
            aa.append(aa_temp)
            pssm_temp = [-float(i) for i in temp]
            pssm.append([pssm_temp[k] for k in idx_res])
    return aa, pssm

def get_phys7(aa):
    phys_dic = {'A': [-0.350, -0.680, -0.677, -0.171, -0.170, 0.900, -0.476],
                            'C': [-0.140, -0.329, -0.359, 0.508, -0.114, -0.652, 0.476],
                            'D': [-0.213, -0.417, -0.281, -0.767, -0.900, -0.155, -0.635],
                            'E': [-0.230, -0.241, -0.058, -0.696, -0.868, 0.900, -0.582],
                            'F': [ 0.363, 0.373, 0.412, 0.646, -0.272, 0.155, 0.318],
                            'G': [-0.900, -0.900, -0.900, -0.342, -0.179, -0.900, -0.900],
                            'H': [ 0.384, 0.110, 0.138, -0.271, 0.195, -0.031, -0.106],
                            'I': [ 0.900, -0.066, -0.009, 0.652, -0.186, 0.155, 0.688],
                            'K': [-0.088, 0.066, 0.163, -0.889, 0.727, 0.279, -0.265],
                            'L': [ 0.213, -0.066, -0.009, 0.596, -0.186, 0.714, -0.053],
                            'M': [ 0.110, 0.066, 0.087, 0.337, -0.262, 0.652, -0.001],
                            'N': [-0.213, -0.329, -0.243, -0.674, -0.075, -0.403, -0.529],
                            'P': [ 0.247, -0.900, -0.294, 0.055, -0.010, -0.900, 0.106],
                            'Q': [-0.230, -0.110, -0.020, -0.464, -0.276, 0.528, -0.371],
                            'R': [ 0.105, 0.373, 0.466, -0.900, 0.900, 0.528, -0.371],
                            'S': [-0.337, -0.637, -0.544, -0.364, -0.265, -0.466, -0.212],
                            'T': [ 0.402, -0.417, -0.321, -0.199, -0.288, -0.403, 0.212],
                            'V': [ 0.677, -0.285, -0.232, 0.331, -0.191, -0.031, 0.900],
                            'W': [ 0.479, 0.900, 0.900, 0.900, -0.209, 0.279, 0.529],
                            'Y': [ 0.363, 0.417, 0.541, 0.188, -0.274, -0.155, 0.476]}
    phys = [phys_dic.get(i, phys_dic['A']) for i in aa]
    return phys

def window(feat, winsize=8):
    feat = np.array(feat)
    output = np.concatenate([np.vstack([feat[0]]*winsize), feat])
    output = np.concatenate([output, np.vstack([feat[-1]]*winsize)])
    output = [np.ndarray.flatten(output[i:i+2*winsize+1]).T for i in range(0,feat.shape[0])]
    return output

def window_data(*feature_types):
    n = len(feature_types[0])
    features = np.empty([n,0])
    for feature_type in feature_types:
        test = np.array(window(feature_type))
        features = np.concatenate((features, test), axis=1)
    return features

def sigmoid(input):
    output = 1 / (1 + np.exp(-input))
    return(output)

def nn_feedforward(nn, input):
    input = np.matrix(input)
    num_layers = nn['n'][0][0][0][0]
    num_input = input.shape[0]
    x = input
    for i in range(1,num_layers-1):
        W = nn['W'][0][0][0][i-1].T
        temp_size = x.shape[0]
        b = np.ones((temp_size,1))
        x = np.concatenate((b, x),axis=1)
        xw=np.dot(x, W)
        x = sigmoid(xw)
    W = nn['W'][0][0][0][-1].T
    b = np.ones((x.shape[0],1))
    x = np.concatenate((b, x),axis=1)
    output =np.dot(x, W)# x*W
    pred = sigmoid(output)
    return pred

dict_ASA0 = dict(zip("ACDEFGHIKLMNPQRSTVWY",
                    (115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
        185, 160, 145, 180, 225, 115, 140, 155, 255, 230)))
def run_iter(dict_nn, input_feature0, aa, ofile):
    SS_order = ('C' 'E' 'H')
    list1 = ('SS', 'ASA', 'TTPP')
    list_res1 = []
    for x in list1:
        nn = dict_nn[x]
        norm_max = nn['high'][0][0][0]
        norm_min = nn['low'][0][0][0]
        input_feature1 = (input_feature0 - np.tile(norm_min, (input_feature0.shape[0], 1))) / np.tile((norm_max - norm_min), (input_feature0.shape[0],1))
        r1 = nn_feedforward(nn, input_feature1)
        list_res1.append(r1)
    pred_ss_1, pred_asa_1, pred_ttpp_1 = list_res1
    SS_1 = [SS_order[i.tolist()[0][0]] for i in np.argmax(pred_ss_1,1)]
    pred_ttpp_1_denorm = (pred_ttpp_1 - 0.5) * 2
    theta = np.degrees(np.arctan2(pred_ttpp_1_denorm[:,0], pred_ttpp_1_denorm[:,2]))
    tau = np.degrees(np.arctan2(pred_ttpp_1_denorm[:,1], pred_ttpp_1_denorm[:,3]))
    phi = np.degrees(np.arctan2(pred_ttpp_1_denorm[:,4], pred_ttpp_1_denorm[:,6]))
    psi = np.degrees(np.arctan2(pred_ttpp_1_denorm[:,5], pred_ttpp_1_denorm[:,7]))
    if ofile == 'NULL':
        return SS_1, pred_ss_1, pred_asa_1, theta, tau, phi, psi
    fp = open(ofile, 'w')
    fp.write('#\tAA\tSS\tASA\tPhi\tPsi\tTheta(i-1=>i+1)\tTau(i-2=>i+1)\tP(C)\tP(E)\tP(H)')
    fp.write('\n')
    for ind, x in enumerate(aa):
        asa = pred_asa_1[ind] * dict_ASA0.get(x, dict_ASA0['A'])

        fp.write(('%i\t%c\t%c\t%5.1f' + '\t%6.1f'*4 + '\t%.3f'*3) % (ind+1, x, SS_1[ind], asa, phi[ind], psi[ind], theta[ind], tau[ind], pred_ss_1[ind,0], pred_ss_1[ind,1], pred_ss_1[ind,2]))
        fp.write('\n')
    fp.close()
    return SS_1, pred_ss_1, pred_asa_1, theta, tau, phi, psi

def mainss(list_params, pssm_file, outfile, out_suffix):
    basenm = os.path.basename(pssm_file)
    if basenm.endswith('.pssm'): basenm = basenm[:-5]
    elif basenm.endswith('.mat'): basenm = basenm[:-4]
    outfile0 = '%s%s.%s' % (outfile, basenm, out_suffix)
    aa, pssm = read_pssm(pssm_file)
    pred1(list_params, aa, pssm, outfile0)
def load_NN(nn_filename):
    return np.load(nn_filename, mmap_mode=None, allow_pickle=True, fix_imports=True, encoding="latin1")        # load in the NN mat file.

def pred1(list_params, aa, pssm, outfile0):
    list_nn = list_params
    phys = get_phys7(aa)
    input_feature = window_data(pssm, phys)
    for it1 in (1, 2, 3):
        ofile = outfile0
        if it1<3: ofile = 'NULL'
        dict_nn = list_nn[it1-1]
        res1 = run_iter(dict_nn, input_feature, aa, ofile)
        if it1 == 3: break
        SS_1, pred_ss_1, pred_asa_1, theta, tau, phi, psi = res1
        tt_input = np.sin(np.concatenate((np.radians(theta), np.radians(tau)), axis=1))/2 + 0.5
        tt_input = np.concatenate((tt_input, np.cos(np.concatenate((np.radians(theta), np.radians(tau)), axis=1))/2 + 0.5), axis=1)
        pp_input = np.sin(np.concatenate((np.radians(phi), np.radians(psi)), axis=1))/2 + 0.5
        pp_input = np.concatenate((pp_input, np.cos(np.concatenate((np.radians(phi), np.radians(psi)), axis=1))/2 + 0.5), axis=1)
        ttpp_input = np.concatenate((tt_input, pp_input), axis=1)
        input_feature = window_data(pssm, phys, pred_ss_1, pred_asa_1, ttpp_input)
    return

def Splitpp(f):
     id = []
     seq = []
     pos = []
     site = []
     st = ''
     with open('../../'+f,'r') as fout:
        for line in fout:
             line=line.rstrip()
             i=i+1
             if line.startswith('>') or '|' in line:
                 if i!=1:
                     st=st.upper()
                     st=st.replace('U','*')
                     seq.append(st)
                     st=''
                 if '|'in line:
                     id.append(line.split('|')[1])
                 elif ' 'in line:
                     id.append(line.split('>')[1].split(' ')[0])
                 elif '\t'in line:
                     id.append(line.split('>')[1].split('\t')[0])
                 else:
                     id.append(line.split('>')[1])
             else:
                 st=st+line
     st=st.upper()
     st=st.replace('U','*')
     seq.append(st)
     for i,k in enumerate(seq):
           po=find_all_index(k,'K')
           pos.append(po)
           sit=[]
           for j,p in enumerate(po):
               if p>=30 and (len(k)-p-1)<30:
                   s=k[p-30:]
                   for i in range(30-(len(k)-p-1)):
                      s=s+"*"
               elif p<30 and (len(k)-p-1)<30:
                   s=k[:]
                   for i in range(30-(len(k)-p-1)):
                      s=s+"*"
                   for i in range(30-(p)):
                      s="*"+s
               elif p<30 and (len(k)-p-1)>=30:
                   s=k[:p+31]
                   for i in range(30-(p)):
                      s="*"+s
               else:s=k[p-30:p+31]
               sit.append(s)
           site.append(sit)
     list0,list30,list20,list10,pid,ps=[],[],[],[],[],[]
     dic=f.split('.')[0][9:]
     if not os.path.exists('123/'+dic):
        os.makedirs('123/'+dic)
     os.chdir('123/'+dic)
     if not os.path.exists('/data/www/sumo/predict/asa_result'):
        os.makedirs('/data/www/sumo/predict/asa_result')
     if not os.path.exists('/data/www/sumo/predict/ZFilepathPSSM'):
        os.makedirs('/data/www/sumo/predict/ZFilepathPSSM')
     if not os.path.exists('/data/www/sumo/predict/ZFilepathresult'):
        os.makedirs('/data/www/sumo/predict/ZFilepathresult')
     with open('/data/www/sumo/predict/peptide.txt', 'w') as fout:
        for i, d in enumerate(id):
            for j, s in enumerate(site[i]):
                list30.append(s)
                list20.append(s[10:-10])
                list10.append(s[20:-20])
                pid.append(d)
                ps.append(pos[i][j] + 1)
                fout.write(s + '\t' + d + '\t' + str(pos[i][j] + 1)  + '\n')
     list0.append(list30)
     list0.append(list20)
     list0.append(list10)
     os.chdir('../../')
     return list0,pid,ps,dic

def acf(list30):
    global acl,aal,acn,aan,acsl,aasl
    pos =DataFrame(list30)
    all_ = pos
    ll,j=0,0
    na={}
    vvl = []
    with open('/data/www/sumo/predict/top10.txt', 'r') as fout:
       line=fout.readline()
       line=line.rstrip()[5:]
       aalist=line.split('\t')
       for line in fout:
           ll=ll+1
           line=line.rstrip()
           na[line.split('\t')[0]]=line.split('\t')[1:]
           vvl.append(line.split('\t')[0])

    def doc2num1(ss):
        sss,AAindex_Encode=[],[]
        ss=ss.replace('*','0')
        ss=ss.replace('X','0')
        ss=ss.replace('B','0')
        ss=ss.replace('U','0')
        for k in vvl:
            s=list(ss)
            for i,ii in enumerate(aalist):
                s = [(na[k][i]) if x == ii else x for x in s]
            s=[float(x) for x in s]
            sss = sss+s
        return sss
    def doc2num(ss):
        AAindex_Encode=[]
        ss=ss.replace('*','0')
        ss=ss.replace('X','0')
        ss=ss.replace('B','0')
        ss=ss.replace('U','0')
        for k in vvl:
            s=list(ss)
            for i,ii in enumerate(aalist):
                s = [(na[k][i]) if x == ii else x for x in s]
            s=[float(x) for x in s]
            AAindex_Encode.append(s)
        ACF_Encode=np.zeros((np.array(AAindex_Encode).shape[0], np.array(AAindex_Encode).shape[1]))
        for i,seq in enumerate(AAindex_Encode):
           for k,kv in enumerate(ss):
            sumValue=0
            for j in range(0,len(seq)-k):
               singleValue=seq[j]*seq[j+k]
               sumValue=sumValue+singleValue
            ACF_Encode[i][k]=round(sumValue/(len(seq)-k),2)
        ACF_Encode=ACF_Encode.flatten().tolist()
        return ACF_Encode
    all_['doc2num']= all_[0].apply(lambda ss: doc2num(ss))
    all_['doc2num1'] = all_[0].apply(lambda ss: doc2num1(ss))
    xy = np.array(list(all_['doc2num']), dtype=np.float)
    x2 = xy.tolist()
    x = np.array(list(all_['doc2num1']),dtype=np.float)
    x3=x.tolist()
    acl,aal=x2,x3
    return acl,aal

def pssm1(list10,dic,protname):
    global ppf
    ppf={}
    pnlist = list10
    with open(dic+'/Protein.txt','w') as fout:
            for it,iter in enumerate(pnlist):
                fout.write('>sp|'+str(protname[it])+'|'+'\n')
                fout.write(iter+"\n")
    ProSeq = dic+"/Protein.txt"
    OutDir = dic+'/ZFilepathresult'
    PSSMDir=dic+'/ZFilepathPSSM'
    DBname = dic+'/ZDB/DB'
    GetPSSM(ProSeq,OutDir,PSSMDir,DBname)
def ss1(list10,dic):
    global sn, asal, ssl,asan, ssn,asasl,sssl
    OldPSMMdir = dic+'/ZFilepathPSSM'
    origin_outfile = dic+'/asa_result'
    detial_pssm = OldPSMMdir + '/'
    outfile = origin_outfile + '/'
    if not os.path.exists(outfile):
        os.makedirs(outfile)
    next_listfile = os.listdir(detial_pssm)
    for i, pssm_file in enumerate(next_listfile):
        if os.path.exists(outfile+pssm_file.split('.')[0]+'.txt'):continue
        nndir = r'/data/www/sumo/predict/SS/'
        dict1_nn = load_NN(nndir + 'pp1.npz')
        dict2_nn = load_NN(nndir + 'pp2.npz')
        dict3_nn = load_NN(nndir + 'pp3.npz')
        list_nn = (dict1_nn, dict2_nn, dict3_nn)
        mainss(list_nn, detial_pssm + '/' + pssm_file, outfile, r'txt')
    sn, asal, ssl,btal,namesout =ss(dic)
    return asal, ssl,btal,namesout
def ss(dic):

    OldPSMMdir =dic+ '/asa_result'#输出
    FinPSSM, FinPSSM1, FinPSSM2,num,namesout = psStandardpPSSM(OldPSMMdir)
    xy = np.array(list(FinPSSM), dtype=np.float)
    x2 = xy.tolist()
    xy1 = np.array(list(FinPSSM1), dtype=np.float)
    x3 = xy1.tolist()
    xy2 = np.array(list(FinPSSM2), dtype=np.float)
    x4 = xy2.tolist()
    return num,x2,x3,x4,namesout

#氨基酸对组成 0、1、2、3
def kmors(list10,km,m='l'):
    global kmn
    pos =DataFrame(list10)
    all_ = pos
    aalist=[]
    if m=='d':
       ablist = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
    else:ablist = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','X','Y']
    for aa in ablist:
         for bb in ablist:
             aalist.append(aa+bb)
    def doc2num(s):
        ssss=[]
        for k in range(km):
            alist=[]
            for i,a in enumerate(s):
                if i+k+1<len(s):
                    alist.append(a+s[i+k+1])
                else:continue
            ss = [float(alist.count(i)) for i in aalist ]
            ssss = ssss+ss[:]
        return list(ssss)
    all_['doc2num'] = all_[0].apply(lambda s: doc2num(s))
    x = np.array(list(all_['doc2num']), dtype=np.int)
    x2 = x.tolist()
    return x2

#氨基酸组成
def aac1(list10):
    pos =DataFrame(list10)
    all_ = pos
    aalist = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
    ll=len(list10[0])
    def doc2num(s):
        s = [float(s.count(i)/ll) for i in aalist ]
        s = s[:]
        return list(s)
    all_['doc2num'] = all_[0].apply(lambda s: doc2num(s))

    x = np.array(list(all_['doc2num']),dtype=np.float)
    x2 = x.tolist()
    return x2

def be1(list10,d):
    pos =DataFrame(list10)
    all_ = pos
    abc = Series(range(0,21),index=['K', 'L', 'A', 'E', 'V', 'G', 'S', 'D', 'I', 'T', 'R', '*', 'P', 'Q','N', 'F', 'Y', 'M', 'H', 'C', 'W'])
    if d==0:abc = Series(range(0,22),index=['K', 'L', 'A', 'E', 'V', 'G', 'S', 'D', 'I', 'T', 'R', '*', 'P', 'Q','N', 'F', 'Y', 'M', 'H', 'C', 'W','U'])
    abc[:] = range(len(abc))
    word_set = set(abc.index)
    def doc2num(s):
        s=s.replace('X','*')
        s=s.replace('B','*')
        s = [i for i in s if i in word_set]
        return list(abc[s])
    all_['doc2num'] = all_[0].apply(lambda s: doc2num(s))

    x = np.array(list(all_['doc2num']), dtype=np.int)
    gen_matrix = lambda z:((np_utils.to_categorical(z, len(abc)).flatten()))
    def data_generator(data,  batch_size):
        batches = [range(batch_size*i, min(len(data), batch_size*(i+1))) for i in range (int(len(data)/batch_size+1))]
        while True:
            for i in batches:
                xx = np.array(list(map(gen_matrix, data[i])))
            return (xx)
    x=data_generator(x[:], len(x)+1)
    bin=x.tolist()
    return bin

#list:[k-mer]
def gps(list):
    global gpn
    def generateMMData(querylist, plist, pls_weight, mm_weight, loo=True, positive=False):
        gp = GpsPredictor(plist, pls_weight, mm_weight)

        d = []

        for query_peptide in querylist:

            d.append(gp.generateMMdata(query_peptide, loo).tolist() )
        return d

    mm_weight = readweight('/data/www/sumo/predict/BLOSUM62R.txt')  # 1th is intercept

    ll=len(list[0])
    gpn = generateMMData(list, list, np.repeat(1, ll), mm_weight, loo=False, positive=False)
    return gpn
def seq2num(seqlist):
    out = []
    transdic = {'A': 8, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 0, 'L': 9, 'M': 10,
                'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, '*': 20}
    for seq in seqlist:
        seq = seq.replace('U', '*').replace('X', '*')
        vec = [transdic[i] for i in seq]
        out.append(vec)
    out = np.array(out)
    return out
class GpsPredictor(object):
    def __init__(self, plist, pls_weight, mm_weight):
        self.alist = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y',
                      'V', 'B', 'Z', 'X', '*']
        self.plist = plist
        self.pls_weight = np.array(pls_weight).flatten()
        self.mm_weight = np.array(mm_weight).flatten()

        self.__count_matrix = self._plist_index()
        self.__mm_matrix, self.__mm_intercept = self._mmweight2matrix()

    def predict(self, query_peptide, loo=False):
        count_clone = self.__count_matrix * len(self.plist)
        matrix = np.zeros_like(self.__count_matrix)
        for i, a in enumerate(query_peptide):
            if a not in self.alist: a = 'C'
            if loo: count_clone[i, self.alist.index(a)] -= 1
            matrix[i, :] = self.__mm_matrix[self.alist.index(a), :]
        rm_num = 1 if loo else 0
        pls_count_matrix = (count_clone.T * self.pls_weight).T / (len(self.plist) - rm_num)
        return np.sum(matrix * pls_count_matrix) + self.__mm_intercept

    def generatePLSdata(self, query_peptide, loo=False):
        count_clone = self.__count_matrix * len(self.plist)
        matrix = np.zeros_like(count_clone)
        for i, a in enumerate(query_peptide):
            if a not in self.alist: a = 'C'
            if loo:
                count_clone[i, self.alist.index(a)] -= 1

            matrix[i, :] = self.__mm_matrix[self.alist.index(a), :]
        rm_num = 1 if loo else 0
        count_clone = (count_clone.T * self.pls_weight).T
        return np.sum(matrix * count_clone / (len(self.plist) - rm_num), 1)

    def generateMMdata(self, query_peptide, loo=False):
        count_clone = self.__count_matrix * len(self.plist)

        indicator_matrix = np.zeros_like(count_clone)
        for i, a in enumerate(query_peptide):
            if a not in self.alist: a = 'C'
            if loo: count_clone[i, self.alist.index(a)] -= 1
            indicator_matrix[i, self.alist.index(a)] = 1

        rm_num = 1 if loo else 0

        count_clone /= (len(self.plist) - rm_num)

        pls_count_matrix = (count_clone.T * self.pls_weight).T

        m = np.dot(indicator_matrix.T, pls_count_matrix) * self.__mm_matrix

        m += m.T

        np.fill_diagonal(m, np.diag(m) / float(2))

        iu1 = np.triu_indices(m.shape[0])

        return m[iu1]

    def getcutoff(self, randompeplist, sp=[0.98, 0.95, 0.85]):
        rand_scores = sorted([self.predict(p) for p in randompeplist])
        cutoffs = np.zeros(len(sp))
        for i, s in enumerate(sp):
            index = np.floor(len(rand_scores) * s).astype(int)
            cutoffs[i] = rand_scores[index]
        return cutoffs

    def _plist_index(self):
        n, m = len(self.plist[0]), len(self.alist)
        count_matrix = np.zeros((n, m))
        for i in range(n):
            for p in self.plist:
                count_matrix[i][self.alist.index(p[i])] += 1
        return count_matrix / float(len(self.plist))

    def _mmweight2matrix(self):
        aalist = self.getaalist()
        mm_matrix = np.zeros((len(self.alist), len(self.alist)))
        for n, d in enumerate(aalist):

            value = self.mm_weight[n + 1]  # mm weight contain intercept
            i, j = self.alist.index(d[0]), self.alist.index(d[1])
            mm_matrix[i, j] = value
            mm_matrix[j, i] = value
        return mm_matrix, self.mm_weight[0]

    def getaalist(self):
        aa = [self.alist[i] + self.alist[j] for i in range(len(self.alist)) for j in range(i, len(self.alist))]
        return aa

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
        outdic = self.outdic + '/' + self.kind
        namelist = list(pepdic.keys())
        peplist = list(pepdic.values())

        ACF,AAindex=acf(peplist)
        d=1
        binary=be1(peplist,d)
        gpn=gps(peplist)
        km=1
        CKSAAPs=kmors(peplist,km,m='l')

        PseAAC=aac1(peplist)
        pssm1(peplist,"/data/www/sumo/predict/",namelist)
        ASA, SS, BTA, namesout = ss1(peplist, "/data/www/sumo/predict/")
        bas_feat={}
        for basname in pepdic.keys():
            pssm=open("/data/www/sumo/predict/ZFilepathPSSM/%s.pssm"%(basname),'r').readlines()[3:-6]
            bas_feat[basname] = []
            for line in pssm:
                if line.startswith(' '):
                    dat=[i for i in line.strip().split(' ')[2:] if i!='']
                    dat=dat[:20]
                    bas_feat[basname]+=dat

        pssmframe=pd.DataFrame.from_dict(bas_feat,orient='index')
        pssmframe.columns=['pssm~'+str(i) for i in pssmframe.columns]
        pssmframe=np.array(pssmframe.values).astype(int)

        transdata = seq2num(peplist)
        modelpath = self.modelfo + '/' + self.kind + '/'

        features = ['ACF', 'AAindex', 'binary', 'gps', 'CKSAAPs', 'PseAAC', 'ASA', 'SS', 'BTA', 'pssm', 'transformer']
        lisarr=[ACF, AAindex, binary, gpn, CKSAAPs, PseAAC, ASA, SS, BTA, pssmframe]

        scores = np.zeros((len(peplist), 1))

        for loc, feat in enumerate(features):

            if feat == 'transformer':
                feat = 'transformer_feat'
                model = load_model(modelpath + feat + '.model')
                outfeat = model.predict(transdata, batch_size=10)

            else:
                model = load_model(modelpath + feat + '.model')
                outfeat = model.predict(np.array(lisarr[loc]))
            outfeat = np.array(outfeat)[:, 1].reshape(-1, 1)
            scores = np.hstack((scores, outfeat))

        scores = np.around(scores[:, 1:],4)
        clfmodel=pickle.load(open(modelpath+'classifier.clf','rb'))

        result = clfmodel.predict_proba(scores)[:, 1]
        scores=[round(i,4) for i in list(result)]

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
        asadic={'A':93.7,'L':173.7,'R':250.4,'K':215.2,'N':146.3,'M':197.6,'D':142.6,'F':228.6,'C':135.2,'P':0,'Q':77.7,'S':109.5,'E':182.9,'T':142.1,'G':52.6,'W':271.6,'H':188.1,'Y':239.9,'I':182.2,'V':157.2}
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
        wholeframe = outframe[['ID', 'Position', 'Peptide', 'Score', 'Cut-off', 'Type', 'link', 'aaindex', 'uniprot']]
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

        wholeframe = wholeframe[['ID', 'Position', 'Peptide', 'Score', 'Cut-off', 'Type', 'link', 'Source', 'aaindex', 'ppi']]
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
        '''
        :param func: 可调用的对象
        :param args: 可调用对象的参数
        '''
        Thread.__init__(self)
        self.args = args
        self.func = func
        self.result = None
    def predict(self):
        self.result=self.func(*self.args)
        return self.result
def func(type,dic):
    global result
    if type == 'Sumoylation':
        sitedic = redfas.readfasta_site(dic)
        if sitedic:
            result1, outframe = redfas.getfeat(dic, sitedic, sumoylation_cutoffdic[sumotion], type)

        else:
            result1 = pd.DataFrame()
            outframe = pd.DataFrame()
        plusdic = {}
        for k, v in countdic.items():
            if k.split('~')[0] + '~site' not in countdic.keys():
                plusdic[k.split('~')[0] + '~site'] = '0;' + str(len(outframe))
        countdic.update(plusdic)
        return result1
    elif type == 'SUMO-interaction':
        sitedic = redfas.readfasta_SIM(dic)
        if sitedic:
            result2, outframe = redfas.getfeat(dic, sitedic, SIM_cutoffdic[sim], type)
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
    if begkind == 'Sumoylation':
        result['count'] = result['ID'].apply(lambda x: countdic[x + '~site'] + '*,;')
    elif begkind == 'SUMO-interaction':
        result['count'] = result['ID'].apply(lambda x: ',;*' + countdic[x + '~sim'])
    else:
        result['count'] = result['ID'].apply(lambda x: countdic[x + '~site'] + '*' + countdic[x + '~sim'])
    return result

'''3s'''
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
from Bio import SeqIO

sumotion=sys.argv[1]
sim=sys.argv[2]
begkind=sys.argv[3]
inputfile=sys.argv[4]
outdic=sys.argv[5]
species=sys.argv[6]
modelfo="/data/www/sumo/predict/models/"
sumoylation_cutoffdic={'All':0,'High':0.75,'Medium':0.55,'Low':0.3}
SIM_cutoffdic = {'All': 0, 'High': 0.50, 'Medium': 0.45, 'Low': 0.40}
kinddic = {'Both': ['Sumoylation', 'SUMO-interaction'], 'Sumoylation': ['Sumoylation'],
           'SUMO-interaction': ['SUMO-interaction']}
wholeframe = pd.DataFrame()
redfas = Predict(inputfile, begkind, outdic, modelfo)
protdic = redfas.protein()
cplm = pd.read_csv("/data/www/sumo/predict/total_0613.txt", sep='\t', index_col=0, header=None)
sitedic = {}
needdic={}
uniprotdic={}
ppi=pd.read_csv("/data/www/sumo/predict/allppi_species.txt",sep='\t',header=None)
if species!='all':
    ppi = ppi[ppi[2] == species]
    uniprotframe = pd.read_csv("/data/www/sumo/webcomp/%s.txt" % (species), sep='\t', header=None)
else:
    uniprotframe=pd.DataFrame()
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





