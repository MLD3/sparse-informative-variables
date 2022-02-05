#COMMANDS FOR ANALYSES FROM PAPER:

#MUST USE THIS VENV:
#cd ~; source venv/bin/activate;cd /data3/interns/postohio/;export CUBLAS_WORKSPACE_CONFIG=:4096:2;export CUDA_VISIBLE_DEVICES=

#Main analyses and ablations:
#for a in simglu ohio;do for x in 0 1 2 3 4 5 6 7 8 9 10 11;do python SIV.py $a $x bo noic;python SIV.py $a $x bo DIRECT; for b in bo SB SIVFIRST SIVLAST z nogate norestrict nodirb;do python SIV.py $a $x $b;done; done;done

#missingness analysis:
#x=9 ; for m in 0 .1 .2 .3 .4 .5; do python SIV.py simglu $x $m bo miss;python SIV.py simglu $x $m miss;done;


import os,datetime
import sys
import matplotlib.pyplot as plt
import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
import joblib
import numpy as np
import pandas as pd
import shutil 
from scipy.ndimage import gaussian_filter1d as gaus
torch.autograd.set_detect_anomaly(True)
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   

def inarg(i,o):
	if i in sys.argv:
		return True,o+'-'+i
	return False,o

#-----------------------------------------------------------------general hyperparameters
BATCHSIZE = 500
SEED=1
nv=6

#All of the backcasts used for ensembling
backcastopts=[24]#[12,18,24,30,36,42]

#prediction horizon
horizon=6#int(sys.argv[1])


PERSUB=False

GOLONG=True
LONGMIN=25
PATIENCE=10
if 'simglu' in sys.argv:
	LONGMIN=500
	PATIENCE=50

SEED=1
if 'seed' in sys.argv:
	SEED=int(sys.argv[2])
	outstr+='.S'+sys.argv[2]

#-----------------------------------------------------------------Data set selection section

#Data set options)
if 'ohio' in sys.argv:
	subjects=['559','563','570','575','588','591','540','544','552','567','584','596']
	outstr='OHIO.'+subjects[int(sys.argv[2])]
	FULL=True
	subjects=[subjects[int(sys.argv[2])]]

SIM=False
SIMGLU=False
datalen=10
if 'simglu' in sys.argv:
	SIM=True
	nv=5
	SIMGLU=True
	missval=0
	
	subs=['adult#001','adult#002','adult#003','adult#004','adult#005','adult#006','adult#007','adult#008','adult#009','adult#010','child#001','child#002','child#003','child#004','child#005','child#006','child#007','child#008','child#009','child#010','adolescent#001','adolescent#002','adolescent#003','adolescent#004','adolescent#005','adolescent#006','adolescent#007','adolescent#008','adolescent#009','adolescent#010']
	if int(sys.argv[2])>9:
		print('No such subject')
		quit()
	SIMSUB=subs[int(sys.argv[2])]
	outstr='SIM.'+str(SIMSUB)
	if 'miss' in sys.argv:
		missval=float(sys.argv[3])
		outstr+='.miss'+sys.argv[3]

#-----------------------------------------------------------------Data processing options ##################################


if 'PERSUB' in sys.argv:
	PERSUB=True
	outstr+='_PERSUB'

PERPERSUB=False
if 'PERPERSUB' in sys.argv:
	PERSUB=True
	PERPERSUB=True
	outstr+='_PERPERSUB'

if 'ens' in sys.argv:# or not SIMDAT:
	backcastopts=[12,18,24,30,36,42]
	outstr+='ENS'



DOTHESCALE=True
SCALEVAL=400
if SIM:
	atempo=np.array(joblib.load('/data3/interns/postohio/NOSNACKsimsubs_X5BIGDELAY/'+SIMSUB+'_1.pkl')[:datalen])
	SCALEVAL=np.max(atempo[:,:,0])
	SCALECARB=np.max(atempo[:,:,2])
	SCALEBOL=np.max(atempo[:,:,1])
	clist=atempo[:,:,2][atempo[:,:,2]>0]
else:
	atempo=joblib.load('/data3/interns/postohio/allohiodata/'+subjects[0]+'.train.pkl')
	gtempo=np.asarray(atempo['glucose'])#/400
	ctempo=np.asarray(atempo['carbs'])
	dtempo=np.asarray(atempo['dose'])
	SCALEVAL=np.nanmax(gtempo)
	SCALECARB=np.nanmax(ctempo)
	SCALEBOL=np.nanmax(dtempo)
	ctempo=ctempo[~np.isnan(ctempo)]
	clist=ctempo[ctempo>0]
CARB50=np.percentile(clist/SCALECARB,50)
CARB75=np.percentile(clist/SCALECARB,75)
CARB90=np.percentile(clist/SCALECARB,90)

DOSMOOTH=False
NOMISS=True
nocf,outstr=inarg('nocf',outstr)
CF=not nocf

#################################### Baseline Options


SUBSAMPLE=0
if 'SIVFIRST' in sys.argv:
	outstr+='_SIVFIRST'
	SUBSAMPLE=1
if 'SIVLAST' in sys.argv:
	outstr+='_SIVLAST'
	SUBSAMPLE=2



#Only do NBEATS-> baseline network
BEATSONLY=False
if 'bo' in sys.argv or (SUBSAMPLE>0 and not 'NOB' in sys.argv):
	BEATSONLY=True
	outstr+='_BEATSONLY'


DOEVAL= 'eval' in sys.argv
MEANOUT=False
LASTOUT=False
if 'MEANOUT' in sys.argv:
	DOEVAL=True
	MEANOUT=True
if 'LASTOUT' in sys.argv:
	DOEVAL=True
	MEANOUT=True
	LASTOUT=True





#-----------------------------------------------------------------Train/test options regarding presense of SIV in input/output

NOINSCARBS=False
if 'noic' in sys.argv:
	NOINSCARBS=True
	outstr+='_NOINSCARBS'
	

TRAINICONLY=False
TRAINNOICONLY=False
TESTICONLY=False
TESTNOICONLY=False




#-----------------------------------------------------------------Model development options


GIVMAINIC=False
nomainic,outstr=inarg('nomainic',outstr)
if not nomainic:
	GIVMAINIC=True
RESTRICT=True
lrdec=1#float(sys.argv[3])
wddec=1#float(sys.argv[4])
lstmsize=1
lstmlay=2
NOGATE,outstr=inarg('nogate',outstr)
norestrict,outstr=inarg('norestrict',outstr)
if norestrict:
	RESTRICT=False


nogateresdir,outstr=inarg('SB',outstr)

if nogateresdir:
	NOGATE=True
	RESTRICT=False



#Primary method has triple capacity for 1/3 of the samples = 2/3 + 1/3*3 =1.67x
if BEATSONLY: 
	PATIENCE*=1.67
	LONGMIN*=1.67




if NOINSCARBS:
	if (not BEATSONLY) or (SUBSAMPLE>0):
			if not NOGATE:
				print('redundant IC run')
				quit()


DIRECT,outstr=inarg('DIRECT',outstr)
nodirb,outstr=inarg('nodirb',outstr)
if not BEATSONLY and not nogateresdir and not nodirb:
	DIRECT=True


#################################### MAIN SECTION ############################################
def main():
	maindir = os.getcwd()+'/'+outstr
	basedir=maindir
	if not DOEVAL:
		os.makedirs(maindir)
	
	
	
	ALLRMSE=[]
	ALLMAE=[]
	
	#set to run all subjects
	loops=[0]
	if PERSUB:
		loops=range(len(subjects)+1)
	for s in loops:
		if s==0:
			sub=99
		else:
			sub=s-1
		if not DOEVAL or True:
			curmodel=0
			if PERSUB:
				if sub==99:
					maindir=basedir+'/ALLSUBS'
					if PERPERSUB:
						continue
				else:
					maindir=basedir+'/'+subjects[sub]
				if not DOEVAL:
					os.makedirs(maindir)
				
			#Training section
			for bc in backcastopts:
				zerodir=basedir+'/ALLSUBS/model'+str(curmodel)
				np.random.seed(SEED)
				torch.manual_seed(SEED)
				train_and_evaluate(curmodel,maindir,horizon,bc,sub,zerodir)
				curmodel=curmodel+1
	
		#final evaluation of ensemble
		#get test data
		train,val,test=makedata(4*6+horizon,sub)
		testgen = ordered_data(BATCHSIZE, 4*6,horizon,test,True)

		#Keep track of total number of evaluated points
		#and total number of each type of event point
		totalpoints=0
		ictotalpoints=0
		ntotalpoints=0
		ictotalpoints50=0
		ictotalpoints75=0
		ictotalpoints90=0
	
		#arrays to store all sorts of losses!
		losses=[]
		rmselosses=[]
		maes=[]
	
		icrmselosses=[]
		icmaes=[]

		icrmselosses=[]
		icmaes=[]

		icrmselosses75=[]
		icmaes75=[]
		icrmselosses50=[]
		icmaes50=[]
		icrmselosses90=[]
		icmaes90=[]
	
		nrmselosses=[]
		nmaes=[]
	
	
		#loop through every batch in training data.
		batch=0
		while(True):
			x,target,done=next(testgen)
			if x.shape[0]<1:
				break
		
			noic= (np.sum(x[:,:,1]+x[:,:,2],1)==0)
			ic= (np.sum(x[:,:,1]+x[:,:,2],1)>0)
			ic50=(np.max(x[:,:,2],1)>CARB50)
			ic75=(np.max(x[:,:,2],1)>CARB75)
			ic90=(np.max(x[:,:,2],1)>CARB90)

			totalpoints = totalpoints+x.shape[0]
		

			#loop through each directory and load predicions
			if not MEANOUT:
				preds=[]
				for f in os.listdir(maindir):
					if f.startswith('model'):
						temp=joblib.load(maindir+'/'+f+'/preds.pkl')
						preds.append(temp[batch])
					#del temp
				#take median
				preds=np.array(preds)
				if len(backcastopts)==1:
					median=temp[batch]
				else:
					median=np.median(preds,axis=0)
			elif LASTOUT:
				median=x[:,-1,0].reshape((-1,1))
			else:
				median=np.mean(x[:,:,0],1).reshape((-1,1))
			#get losses
			if not MEANOUT:
				losses.append(mse_cpu(target, median)*x.shape[0])
		
			rmselosses.append(mse_lastpointonly_cpu(target*SCALEVAL, median*SCALEVAL)*x.shape[0])
			maes.append(mae_lastpointonly_cpu(target*SCALEVAL, median*SCALEVAL)*x.shape[0])
		
			if np.sum(ic)>1:
				icrmselosses.append(mse_lastpointonly_cpu(target[ic,:]*SCALEVAL, median[ic,:]*SCALEVAL)*x[ic,:,:].shape[0])
				icmaes.append(mae_lastpointonly_cpu(target[ic,:]*SCALEVAL, median[ic,:]*SCALEVAL)*x[ic,:,:].shape[0])
				ictotalpoints = ictotalpoints+x[ic,:,:].shape[0]
			if np.sum(ic50)>1:
				icrmselosses50.append(mse_lastpointonly_cpu(target[ic50,:]*SCALEVAL, median[ic50,:]*SCALEVAL)*x[ic50,:,:].shape[0])
				icmaes50.append(mae_lastpointonly_cpu(target[ic50,:]*SCALEVAL, median[ic50,:]*SCALEVAL)*x[ic50,:,:].shape[0])
				ictotalpoints50 = ictotalpoints50+x[ic50,:,:].shape[0]
			if np.sum(ic75)>1:
				icrmselosses75.append(mse_lastpointonly_cpu(target[ic75,:]*SCALEVAL, median[ic75,:]*SCALEVAL)*x[ic75,:,:].shape[0])
				icmaes75.append(mae_lastpointonly_cpu(target[ic75,:]*SCALEVAL, median[ic75,:]*SCALEVAL)*x[ic75,:,:].shape[0])
				ictotalpoints75 = ictotalpoints75+x[ic75,:,:].shape[0]
			if np.sum(ic90)>1:
				icrmselosses90.append(mse_lastpointonly_cpu(target[ic90,:]*SCALEVAL, median[ic90,:]*SCALEVAL)*x[ic90,:,:].shape[0])
				icmaes90.append(mae_lastpointonly_cpu(target[ic90,:]*SCALEVAL, median[ic90,:]*SCALEVAL)*x[ic90,:,:].shape[0])
				ictotalpoints90 = ictotalpoints90+x[ic90,:,:].shape[0]
		
			if np.sum(noic)>1:
				nrmselosses.append(mse_lastpointonly_cpu(target[noic,:]*SCALEVAL, median[noic,:]*SCALEVAL)*x[noic,:].shape[0])
				nmaes.append(mae_lastpointonly_cpu(target[noic,:]*SCALEVAL, median[noic,:]*SCALEVAL)*x[noic,:].shape[0])
				ntotalpoints = ntotalpoints+x[noic,:,:].shape[0]
			batch=batch+1
			if done:
				break
		
		#write final losses
		#MSE for whole window
		if MEANOUT:
			print(str(np.sqrt(np.sum(np.asarray(rmselosses))/totalpoints)))
			print(totalpoints)
			print(ictotalpoints)
			quit()
		t=open(maindir+"/"+str(np.sum(np.asarray(losses))/totalpoints)+".FINALMSEout","w")
	

		#rmse and mae for last point only
		t=open(maindir+"/"+str(np.sqrt(np.sum(np.asarray(rmselosses))/totalpoints))+".FINAL_RMSE_out","w")
		t=open(maindir+"/"+str(np.sum(np.asarray(maes))/totalpoints)+".FINALMAEout","w")

		t=open(maindir+"/"+str(np.sqrt(np.sum(np.asarray(icrmselosses))/ictotalpoints))+".FINAL_RMSE_ICout","w")
		t=open(maindir+"/"+str(np.sum(np.asarray(icmaes))/ictotalpoints)+".FINALMAEICout","w")



		t=open(maindir+"/"+str(np.sqrt(np.sum(np.asarray(icrmselosses50))/ictotalpoints50))+".FINAL_RMSE_IC50out","w")
		t=open(maindir+"/"+str(np.sum(np.asarray(icmaes50))/ictotalpoints50)+".FINALMAEIC50out","w")

		t=open(maindir+"/"+str(np.sqrt(np.sum(np.asarray(icrmselosses75))/ictotalpoints75))+".FINAL_RMSE_IC75out","w")
		t=open(maindir+"/"+str(np.sum(np.asarray(icmaes75))/ictotalpoints75)+".FINALMAEIC75out","w")

		t=open(maindir+"/"+str(np.sqrt(np.sum(np.asarray(icrmselosses90))/ictotalpoints90))+".FINAL_RMSE_IC90out","w")
		t=open(maindir+"/"+str(np.sum(np.asarray(icmaes90))/ictotalpoints90)+".FINALMAEIC90out","w")



	
		t=open(maindir+"/"+str(np.sqrt(np.sum(np.asarray(nrmselosses))/ntotalpoints))+".FINAL_RMSE_ICNOout","w")
		t=open(maindir+"/"+str(np.sum(np.asarray(nmaes))/ntotalpoints)+".FINALMAEICNOout","w")
		
		if sub<99:
			ALLRMSE.append(np.sqrt(np.sum(np.asarray(rmselosses))/totalpoints))
			ALLMAE.append(np.sum(np.asarray(maes))/totalpoints)

	if PERSUB:
		#write final metrics
		t=open(basedir+"/"+str(np.mean(ALLRMSE))+'.RMSE',"w")
		t=open(basedir+"/"+str(np.mean(ALLMAE))+'.MAE',"w")

####################################  TRAINING, AND EVALUATION SECTION ############################################
def train_and_evaluate(curmodel,maindir,forecast_length,backcast_length,sub,basedir):
	mydir = maindir+'/model'+str(curmodel)
	if not DOEVAL:
		os.makedirs(mydir)
	print(mydir)
	
	
	
	pin_memory=True
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	batch_size = BATCHSIZE
	
	train,val,test=makedata(backcast_length+forecast_length,sub)
	traingen = data(batch_size, backcast_length, forecast_length,train,0)
	valgen = data(batch_size, backcast_length, forecast_length,val,0)
	traingenic = data(batch_size, backcast_length, forecast_length,train,2)
	valgenic = data(batch_size, backcast_length, forecast_length,val,2)
	traingennoic = data(batch_size, backcast_length, forecast_length,train,1)
	valgennoic = data(batch_size, backcast_length, forecast_length,val,1)
	

	testgennoic = data(batch_size, backcast_length, forecast_length,test,1)
	
	

	testgen = ordered_data(batch_size, backcast_length, forecast_length,test)
	valtestgen=ordered_data(batch_size, backcast_length, forecast_length,val)
	net = network(device,backcast_length,forecast_length)
	# lrdec=1
	# if sub!=99:
	# 	lrdec=.01
	# elif SIMPRE:
	# 	lrdec=.1
	optimiser = optim.Adam(net.parameters(),lr=.01*lrdec,weight_decay=.0000001*wddec)#.0002


	if not DOEVAL:
		if SUBSAMPLE==1:
			fit(net, optimiser, traingenic,valgenic,mydir, device,basedir,sub)
			fit(net, optimiser, traingen,valgen,mydir, device,basedir,sub)#,not BEATSONLY)
		elif SUBSAMPLE==2:
			fit(net, optimiser, traingen,valgen,mydir, device,basedir,sub)#
			fit(net, optimiser, traingenic,valgenic,mydir, device,basedir,sub)
		else:
			fit(net, optimiser, traingen,valgen,mydir, device,basedir,sub)

	eval(net, optimiser, valtestgen,mydir,  device,'val')
	eval(net, optimiser, testgen,mydir,  device,'')
	eval(net, optimiser, testgen,mydir,  device,'noic')
	







def fit(net, optimiser, traingen,valgen,mydir,device, basedir,sub):#bonusdata=None):
	losss=mse
	improvepoint=0

	loadnoopt(net, optimiser,basedir)#first load pretrained model
	loadnoopt(net, optimiser,mydir)#if this is the second pass, then load the model.

	
	trains=[]
	vals=[]
	patience=PATIENCE
	if sub<99:
		patience=15
	prevvalloss=np.inf
	unimproved=0


	net.to(device)

	
	loops=5000000
	if 'lil' in sys.argv:
		loops=2
	if 'lillil' in sys.argv:
		loops=0


	for grad_step in range(loops):

		temptrain=[]
		total=0
		while(True):
			optimiser.zero_grad()
			net.train()
			x,target,done=next(traingen)
			if x.shape[0]<1:
				break
				
				
			if x.shape[0]<1:
				break

			forecast,ul=  net(   torch.tensor(x, dtype=torch.float).to(device),torch.tensor(target, dtype=torch.float).to(device))

			loss = losss(forecast.to(device)	, torch.tensor(target, dtype=torch.float).to(device)).to(device)	


			loss.backward()
			temptrain.append(loss.item()*x.shape[0])
			total=total+x.shape[0]
			optimiser.step()
			
			if done:
				break

		
		
		trains.append(np.sum(temptrain)/total)
		print('grad_step = '+str(grad_step)+' loss = '+str(trains[-1]))
		
		
		tempval=[]
		total=0
		while(True):
			with torch.no_grad():
				x,target,done=next(valgen)
				if x.shape[0]<1:
					break

				forecast,ul=  net(   torch.tensor(x, dtype=torch.float).to(device),torch.tensor(target, dtype=torch.float).to(device))		
			
			loss = losss(forecast.to(device)	, torch.tensor(target, dtype=torch.float).to(device)).to(device)				
			total=total+x.shape[0]
			tempval.append(loss.item()*x.shape[0])
			if done:
				break
		vals.append(np.sum(tempval)/total)
		
		print('val loss: '+str(vals[-1]))				
		
		
		if vals[-1]<prevvalloss:
			print('loss improved')
			improvepoint=grad_step
			prevvalloss=vals[-1]
			unimproved=0
			save(net, optimiser, grad_step,mydir)
		else:
			unimproved+=1
			print('loss did not improve for '+str(unimproved)+'th time')
		if (unimproved>patience):
			if GOLONG and grad_step<LONGMIN:
				print('going long')
				continue
			print('Finished.')
			t=open(mydir+"/"+str(improvepoint)+'_ITS',"w")
			break
	plt.plot(range(len(trains)),trains,'k--', range(len(trains)),vals,'r--')
	plt.legend(['train','val'])
	plt.savefig(mydir+"/loss_over_time.png")
	plt.clf()
	del net
			
			
			
def eval(net, optimiser, testgen,mydir,  device,OSTR):
	with torch.no_grad():
		load(net, optimiser,mydir)
		totalpoints=0
		losses=[]
		rmselosses=[]
		preds=[]
		xs=[]
		targs=[]

		while(True):
			x,target,done=next(testgen)
			if x.shape[0]<1:
				break
			if OSTR=='noic':
				x[:,:,1:3]=0
			xs.append(x)
			targs.append(target)
			totalpoints = totalpoints+x.shape[0]
			forecast,ul= net(   torch.tensor(x, dtype=torch.float).to(device),torch.tensor(target, dtype=torch.float).to(device))		
			forecast=forecast.to(device)
			preds.append(forecast.cpu().numpy())

				
			#print(sivback.cpu().numpy())
			losses.append(mse(forecast.to(device)*SCALEVAL, torch.tensor(target*SCALEVAL, dtype=torch.float).to(device)).item()*x.shape[0])
			rmselosses.append(mse_lastpointonly(forecast.to(device)*SCALEVAL, torch.tensor(target*SCALEVAL, dtype=torch.float).to(device)).item()*x.shape[0])
			if done:
				break
		#write final loss
		t=open(mydir+"/"+str(np.sum(np.asarray(losses))/totalpoints)+".test"+OSTR+"MSEout","w")
		t=open(mydir+"/"+str(np.sqrt(np.sum(np.asarray(rmselosses))/totalpoints))+".test"+OSTR+"Rmseout","w")
		if DOEVAL:
			print( str(np.sum(np.asarray(losses))/totalpoints) )
			print( str(np.sqrt(np.sum(np.asarray(rmselosses))/totalpoints)) )

		#dump out predictions to be used in ensembling
		if OSTR=='' or OSTR=='noic':
			joblib.dump(preds,mydir+'/preds'+OSTR+'.pkl')
			joblib.dump(xs,mydir+'/xs'+OSTR+'.pkl')
			joblib.dump(targs,mydir+'/targs'+OSTR+'.pkl')






###################################SAVE AND LOAD FUNCTIONS
def save(model, optimiser, grad_step,mdir):
	torch.save({
		'grad_step': grad_step,
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimiser.state_dict(),
	}, mdir+'/model_out.th')

def load(model, optimiser,mdir):
	if os.path.exists(mdir+'/model_out.th'):
		print('loading '+mdir)
		checkpoint = torch.load(mdir+'/'+'model_out.th')
		model.load_state_dict(checkpoint['model_state_dict'])
		optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
		grad_step = checkpoint['grad_step']

def loadnoopt(model, optimiser,mdir):
	if os.path.exists(mdir+'/model_out.th'):
		print('loading '+mdir)
		checkpoint = torch.load(mdir+'/'+'model_out.th')
		model.load_state_dict(checkpoint['model_state_dict'])






############################################    LOSS FUNCTIONS

def mse(output, target):
	if not SIM:
		output=output[target!=0]
		target=target[target!=0]
	out=torch.mean((output - target)**2)
	return out


	
def mae(output, target):
	output=output[target!=0]
	target=target[target!=0]
	return torch.mean(torch.abs((output - target)/target))

def mse_lastpointonly(output, target):
	output=output[:,-1]
	target=target[:,-1]
	loss = torch.mean((output - target)**2)
	return loss 



def mse_cpu(output, target):
	output=output[target!=0]
	target=target[target!=0]
	return np.mean((output - target)**2)

def mse_lastpointonly_cpu(output, target):
	output=output[:,-1]
	target=target[:,-1]
	loss = np.mean((output - target)**2)
	return loss 
	
def mae_lastpointonly_cpu( target,output):
	output=output[:,-1]
	target=target[:,-1]
	loss = np.mean(np.abs(output - target))
	return loss 
	





	
	
	
####################################  MODEL SECTION  ############################################################################################################  


class Block(nn.Module):

	def __init__(self, units, device, backcast_length, forecast_length):
		super(Block, self).__init__()
		self.backlen=backcast_length
		self.forecast_length=forecast_length
		self.input=nv
		self.device = device
	
		self.units=int(100*lstmsize)
		self.bs=BATCHSIZE
		

		#main encoder network
		self.lstm=nn.LSTM(self.input,self.units, num_layers=lstmlay,batch_first=True,bidirectional=True).to(device)

		
		#main decoder network
		self.dec=nn.LSTM(self.units*2,self.units, num_layers=lstmlay,batch_first=True,bidirectional=True).to(device)

		if BEATSONLY and DIRECT:
			self.dec=nn.LSTM(self.units*2+60,self.units, num_layers=lstmlay,batch_first=True,bidirectional=True).to(device)
		if not BEATSONLY:
			#SIV encoder network
			self.lstmS=nn.LSTM(self.input,self.units, num_layers=2,batch_first=True,bidirectional=True).to(device)
			self.decS=nn.LSTM(self.units*4,self.units*2, num_layers=2,batch_first=True,bidirectional=True).to(device)

			self.lstmSC=nn.LSTM(self.input,self.units, num_layers=2,batch_first=True,bidirectional=True).to(device)
			self.decSC=nn.LSTM(self.units*4,self.units*2, num_layers=2,batch_first=True,bidirectional=True).to(device)
			self.linSC=nn.Linear(self.units *2, 1).to(device)
			
			self.decS=nn.LSTM(self.units*4+30,self.units*2, num_layers=2,batch_first=True,bidirectional=True).to(device)
			self.decSC=nn.LSTM(self.units*4+30,self.units*2, num_layers=2,batch_first=True,bidirectional=True).to(device)
			self.decS=nn.LSTM(self.units*2,self.units, num_layers=2,batch_first=True,bidirectional=True).to(device)
			self.decSC=nn.LSTM(self.units*2,self.units, num_layers=2,batch_first=True,bidirectional=True).to(device)
			if DIRECT:
				self.decS=nn.LSTM(self.units*2+30,self.units, num_layers=2,batch_first=True,bidirectional=True).to(device)
				self.decSC=nn.LSTM(self.units*2+30,self.units, num_layers=2,batch_first=True,bidirectional=True).to(device)
		#output network
		self.lin=nn.Linear(self.units *2, 1).to(device)


		self.to(device)
		
	def forward(self, xt,xorig):
		



		x=xt.clone()
		origbs=x.size()[0]
		if origbs<self.bs:
			x=F.pad(input=x, pad=( 0,0,0,0,0,self.bs-origbs), mode='constant', value=0).to(self.device)
			xorig=F.pad(input=xorig, pad=( 0,0,0,0,0,self.bs-origbs), mode='constant', value=0).to(self.device)




		if not BEATSONLY:
			bothinds=torch.sum(torch.sum(x[:,:,1:3].clone(),1),1)>0
			inds=torch.sum(x[:,:,1],1)>0
			Cinds=torch.sum(x[:,:,2],1)>0
			xin=x.clone()
			if not GIVMAINIC:
				xin[:,:,1:3]=0

			# if RECBACK:
			# 	xin=xint.clone()
		else:
			xin=x.clone()	

		if DIRECT and BEATSONLY and not GIVMAINIC:
			xin[:,:,1:3]=0

		lstm_out, (h_0,c_0) = self.lstm(xin)
		lstm_out=lstm_out[:,-1,:].view((500,1,-1))
		
		outer=torch.zeros(self.bs,self.forecast_length).to(self.device)



		hdec=(torch.zeros(2*lstmlay,self.bs,self.units)).to(self.device)#,
		cdec = (torch.zeros(2*lstmlay,self.bs,self.units)).to(self.device)#
		hdecS=(torch.zeros(4,self.bs,self.units)).to(self.device)#,
		cdecS = (torch.zeros(4,self.bs,self.units)).to(self.device)#,
		hdecSC=(torch.zeros(4,self.bs,self.units)).to(self.device)#,
		cdecSC = (torch.zeros(4,self.bs,self.units)).to(self.device)#,


		if DIRECT:
			sivDIR=torch.zeros((500,30,2)).to(self.device)#,
			sivDIR[:,6:,:]=x[:,:,1:3].clone()
			if not BEATSONLY:
				sivDIRC=sivDIR[:,:,1:2].clone()
				sivDIR=sivDIR[:,:,0:1].clone()
		for f in range(self.forecast_length):
			if DIRECT and BEATSONLY:
				lstm_out=torch.cat((lstm_out.view((500,1,-1)),sivDIR.clone().view(500,1,60)  /.7*torch.mean(lstm_out.view((500,1,-1)),2).view(500,1,1)   ),2)
				for i in range(0,self.backlen+5):
					sivDIR[:,i,:]=sivDIR[:,i+1,:]
			lstm_outxx, (hdec,cdec) = self.dec(lstm_out,(hdec,cdec))
			lstm_outx=lstm_outxx.clone()	
			outer[:,f]=self.lin(lstm_outx.clone()[:,0,:]).view(-1)
			if not BEATSONLY:
				if DIRECT:
					lstm_outS, (hdecS,cdecS) = self.decS(torch.cat((lstm_out,sivDIR.clone().view(500,1,30) /.7*torch.mean(lstm_out.view((500,1,-1)),2).view(500,1,1) ),2),(hdecS,cdecS))
					for i in range(0,self.backlen+5):
						sivDIR[:,i,:]=sivDIR[:,i+1,:]
				else:
					lstm_outS, (hdecS,cdecS) = self.decS(lstm_out,(hdecS,cdecS))		
				lstmotemp=lstm_outx.clone()
				if NOGATE:
					if RESTRICT:
						lstmotemp[:,0,:]=lstm_outx[:,0,:].clone()+F.relu(lstm_outS[:,0,:self.units*2].clone())
					else:
						lstmotemp[:,0,:]=lstm_outx[:,0,:].clone()+lstm_outS[:,0,:self.units*2].clone()
				else:
					if not RESTRICT:
						lstmotemp[inds,0,:]=lstm_outx[inds,0,:].clone()-lstm_outS[inds,0,:self.units*2].clone()
					else:
						lstmotemp[inds,0,:]=lstm_outx[inds,0,:].clone()-F.relu(lstm_outS[inds,0,:self.units*2].clone())
				lstm_outx=lstmotemp.clone()

				if DIRECT:
					lstm_outSC, (hdecSC,cdecSC) = self.decSC(torch.cat((lstm_out,sivDIRC.clone().view(500,1,30) /.7*torch.mean(lstm_out.view((500,1,-1)),2).view(500,1,1) ),2),(hdecSC,cdecSC))
					for i in range(0,self.backlen+5):
						sivDIRC[:,i,:]=sivDIRC[:,i+1,:]
				else:
					lstm_outSC, (hdecSC,cdecSC) = self.decSC(lstm_out.view((500,1,-1)),(hdecSC,cdecSC))		
				lstmotemp=lstm_outx.clone()
				if NOGATE:
					if RESTRICT:
						lstmotemp[:,0,:]=lstm_outx[:,0,:].clone()+F.relu(lstm_outSC[:,0,:self.units*2].clone())
					else:
						lstmotemp[:,0,:]=lstm_outx[:,0,:].clone()+lstm_outSC[:,0,:self.units*2].clone()
				else:
					if not RESTRICT:
						lstmotemp[Cinds,0,:]=lstm_outx[Cinds,0,:].clone()+lstm_outSC[Cinds,0,:self.units*2].clone()
					else:
						lstmotemp[Cinds,0,:]=lstm_outx[Cinds,0,:].clone()+F.relu(lstm_outSC[Cinds,0,:self.units*2].clone())
				lstm_outx=lstmotemp.clone()
				outer[:,f]=self.lin(lstm_outx.clone()[:,0,:]).view(-1)
				lstm_outST=lstm_outS.clone()
				lstm_outSCT=lstm_outSC.clone()


			lstm_out=lstm_outx.clone()
			
		

		outer=outer[:origbs,:]



		return outer,[]












class network(nn.Module):
	def __init__(self,device,backcast_length,forecast_length):
		super(network, self).__init__()
		self.forecast_length = forecast_length
		self.backcast_length = backcast_length
		self.hidden_layer_units = 512
		self.nb_blocks_per_stack = 1
		
		self.device=device

		self.mainblock=Block(self.hidden_layer_units,device, backcast_length, forecast_length).to(device)
		self.to(self.device)
		
  

	def forward(self, x,target):

		xorig=x.clone()
		if CF:
			for f in range(1,x.shape[1]):
				x[:,f,1:3]+=x[:,f-1,1:3]
		
		forecast,ul=self.mainblock(x,xorig)

		return forecast,ul





####################################  DATA GENERATION SECTION  ############################################################################################################  

def makedata(totallength,sub):
	train=[]
	test=[]
	val=[]
	
	

	if SIM:
		print('loading '+SIMSUB)
		a=joblib.load('/data3/interns/postohio/NOSNACKsimsubs_X5BIGDELAY/'+SIMSUB+'_1.pkl')
		

		a=a[:datalen]
		ll=len(a)
		if missval>0:
			np.random.seed(SEED)
		for f in range(ll):
			ff=a[f][:,:3]
			ff[:,0]/=SCALEVAL
			ff[:,1]/=SCALEBOL
			ff[:,2]/=SCALECARB
			if missval>0:
				
				for i in range(ff.shape[0]):
					if not ff[i,2]==0 and not np.isnan(ff[i,2]):
						temppp=np.random.uniform()
						if temppp<missval:
							ff[i,2]=0
						else:
							ff[i,2]=ff[i,2]*(1-missval+np.random.uniform()*missval*2)

			t=np.arange(288)
			ff=np.concatenate((ff,np.zeros([288,2])),1)
			ff[:,-2]=np.sin( t*2*np.pi/288)
			ff[:,-1]=np.cos( t*2*np.pi/288)
			if f<=ll*.7:
				train.append(ff)
			elif f<=ll*.85:
				val.append(ff)
			else:
				test.append(ff[50-(totallength-12-1):,:])

		return train,val,test




	folder='/data3/interns/postohio/allohiodata'
	#first load train data
	stored_trains={}
	storedd={}
	storedc={}
	for f in os.listdir(folder):
		if f.endswith('train.pkl'):
			if not sub==99: 
				if not f[:3]==subjects[sub]:
					continue
			else:
				if not f[:3] in subjects:
					continue
			a=joblib.load(folder+'/'+f)
			g=np.asarray(a['glucose'])#/400
			if DOTHESCALE:
				g/=SCALEVAL
			b=np.asarray(a['basal'])
			d=np.asarray(a['dose'])
			c=np.asarray(a['carbs'])
			
			c[np.isnan(c)]=0
			d[np.isnan(d)]=0
			
			if DOTHESCALE:
				d/=SCALECARB
				c/=SCALEBOL
			
			


			fing=np.asarray(a['finger'])/400.0
			hr=np.asarray(a['hr'])
			gsr=np.asarray(a['gsr'])
			t=np.array(a.index.values)
			t1=np.sin( t*2*np.pi/288)
			t2=np.cos( t*2*np.pi/288)
			miss=(np.isnan(g)).astype(float)
			miss2=(np.isnan(fing)).astype(float)
			x=np.stack((g,d,c,t1,t2,miss,b),axis=1)
			if NOMISS:
				x=np.stack((g,d,c,t1,t2,b),axis=1)
			# if fourvar:
			# 	x=np.stack((g,d,c,b),axis=1)
			ll=x.shape[0]
			train.append(x.copy()[:int(ll*.8),:])
			val.append(x.copy()[int(ll*.8):,:])
			#store to use in test for end
			stored_trains[f]=x.copy()
		
			
	for f in os.listdir(folder):
		if f.endswith('test.pkl'):
			if not sub==99: 
				if not f[:3]==subjects[sub]:
					continue
			else:
				if not f[:3] in subjects:
					continue
			
			print(f[:3])
			a=joblib.load(folder+'/'+f)
			g=np.asarray(a['glucose'])#/400
			if DOTHESCALE:
				g/=SCALEVAL
			b=np.asarray(a['basal'])
			d=np.asarray(a['dose'])
			c=np.asarray(a['carbs'])
			
			
			c[np.isnan(c)]=0
			d[np.isnan(d)]=0
			
			if DOTHESCALE:
				d/=SCALECARB
				c/=SCALEBOL
			


			
			fing=np.asarray(a['finger'])/400.0
			hr=np.asarray(a['hr'])
			gsr=np.asarray(a['gsr'])
			t=np.array(a.index.values)
			miss2=(np.isnan(fing)).astype(float)
			t1=np.sin( t*2*np.pi/288)
			t2=np.cos(t *2*np.pi/288)
			miss=(np.isnan(g)).astype(float)
			x=np.stack((g,d,c,t1,t2,miss,b),axis=1)
			if NOMISS:
				x=np.stack((g,d,c,t1,t2,b),axis=1)
			# if fourvar:
			# 	x=np.stack((g,d,c,b),axis=1)
			t=stored_trains[f.replace('test','train')]
			x=np.concatenate((t[-(totallength-12-1):,:],x),axis=0)
			test.append(x.copy())
	return train,val,test








def data(num_samples, backcast_length, forecast_length, data,ic):
		def get_x_y(ii,ic):  
				temp=data[0]
				done=False
				startnum=0
				if ic==9:
					startnum=checklen
				for s in range(len(data)):
						temp=data[s]
						if len(temp)<backcast_length+ forecast_length+startnum:
								continue
						if ii+startnum<=len(temp)-backcast_length-forecast_length:
								done=True
								break
						ii=ii-(len(temp)-backcast_length-forecast_length-startnum)-1
				if not done:
						return None,None,True
								


				i=ii+startnum
				learn=temp[i:i+backcast_length]
				see=temp[i+backcast_length:i+backcast_length+forecast_length]
				see[np.isnan(see)]=0
				learn[np.isnan(learn)]=0
				origlearn=learn.copy()
				origsee=see.copy()
				if ic==9:
					prev=temp[i-startnum:i,1:3]
					if np.nansum(prev)>0:
						return np.asarray([]),None,False
				if DOSMOOTH:
					l=learn[:,0].copy()
					l[l==0]=np.nan
					l2=l.copy()
					l[-2]=np.nanmean(l2[-2:])
					for ii in range(learn.shape[0]-2):
						l[ii]=np.nanmean(l2[ii:ii+3])
					l[np.isnan(l)]=0
					learn[:,0]=l
					l[learn[:,0]==0]=0

				see=temp[i+backcast_length:i+backcast_length+forecast_length,0]
				if TRAINICONLY or ic==2:
					if np.sum(learn[:,1])+np.sum(learn[:,2])==0:
						return np.asarray([]),None,False
				if TRAINNOICONLY or ic==1:
					if np.sum(learn[:,1])+np.sum(learn[:,2])!=0:
						return np.asarray([]),None,False
				if np.prod(see)==0:
					return np.asarray([]),None,False
				if NOMISS and np.prod(learn[:,0])==0:
					return np.asarray([]),None,False
				if np.sum(learn[:,0])==0:
					return np.asarray([]),None,False
				return learn,see,False
		   
		
		
		def gen():
				done=False
				indices=range(99999999)
				xx = []
				yy = []
				i=0
				added=0
				while(True):
						x, y,done = get_x_y(indices[i],ic)
						i=i+1
						if done or i==len(indices):
								xx=np.array(xx)
								if NOINSCARBS:
									if xx.shape[0]>0:
										xx[:,:,1]=0
										xx[:,:,2]=0
										
										#xx[:,:,-1]=0
								yield xx, np.array(yy),True
								done=False
								xx = []
								yy = []
								if indices[100]==100 and indices[101]==101:
										indices=np.random.permutation(i-1)
								else:
										indices=np.random.permutation(i)
								i=0
								added=0
								continue
						if not x.shape[0]==0:
								xx.append(x)
								yy.append(y)
								added=added+1
								if added%num_samples==0:
										xx=np.array(xx)
										if NOINSCARBS:
											if xx.shape[0]>0:
												xx[:,:,1]=0
												xx[:,:,2]=0
												
												#xx[:,:,-1]=0
										yield xx, np.array(yy),done
										xx = []
										yy = []
		return gen()



def ordered_data(num_samples, backcast_length, forecast_length, dataa,doicanyway=False):
	def get_x_y(i):  
		temp=dataa[0]
		done=False
		for s in range(len(dataa)):
			temp=dataa[s]
			#if this time series is too short, skip it.
			if len(temp)<backcast_length+ forecast_length:
				continue
			#if this index falls within this time series, we can return it
			if i<=len(temp)-backcast_length-forecast_length:
				done=True
				break
			#otherwise subtract this subject's points and keep going.
			i=i-(len(temp)-backcast_length-forecast_length)-1
		#if we're out of data, quit.
		if not done:
			return None,None,True
		learn=temp[i:i+backcast_length]
		see=temp[i+backcast_length:i+backcast_length+forecast_length]
		see[np.isnan(see)]=0
		learn[np.isnan(learn)]=0
		origlearn=learn.copy()
		if DOSMOOTH:
			l=learn[:,0].copy()
			l[l==0]=np.nan
			l2=l.copy()
			l[-2]=np.nanmean(l2[-2:])
			for ii in range(learn.shape[0]-2):
				l[ii]=np.nanmean(l2[ii:ii+3])
			l[np.isnan(l)]=0
			learn[:,0]=l
			l[learn[:,0]==0]=0
			# learn[:,0]=gaus(learn[:,0],1)
			# learn[:,0][origlearn[:,0]==0]=0
		see=temp[i+backcast_length:i+backcast_length+forecast_length,0]
		if TESTICONLY:
			if np.sum(learn[:,1])+np.sum(learn[:,2])==0:
				return np.asarray([]),None,False
		if TESTNOICONLY:
			if np.sum(learn[:,1])+np.sum(learn[:,2])!=0:
				return np.asarray([]),None,False
		#only use data where the point we're trying to predict is there.
		if see[-1]==0:
			return np.asarray([]),None,False
		if NOMISS and (np.prod(learn[:,0])==0 or np.prod(see)==0):
					return np.asarray([]),None,False
		return learn,see,False
	
	
	
	def gen():
		done=False
		xx = []
		yy = []
		i=0
		added=0
		while(True):
			x, y,done = get_x_y(i)
			i=i+1
			if done:
				xx=np.array(xx)
				if NOINSCARBS and (not doicanyway):
					if xx.shape[0]>0:
						xx[:,:,1]=0
						xx[:,:,2]=0
						
						#xx[:,:,-1]=0
				yield xx, np.array(yy),True
				done=False
				xx = []
				yy = []
				i=0
				added=0
				continue
			if not x.shape[0]==0:
				xx.append(x)
				yy.append(y)
				added=added+1
				if added%num_samples==0:
					xx=np.array(xx)
					if NOINSCARBS and (not doicanyway):
						if xx.shape[0]>0:
							xx[:,:,1]=0
							xx[:,:,2]=0
							
							#xx[:,:,-1]=0
					yield xx, np.array(yy),False
					xx = []
					yy = []
	return gen()
	


if __name__ == '__main__':
	main()
	
	
	

