#This code makes result plots and tables.

import os,sys
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os,sys
import pandas as pd
import scipy.stats

#####Inclune ohio as an input to gater results from real data, otherwise run with no inputs###


####THIS SECTION MAKES THE OUTPUT TABLE#######

OHIO='ohio' in sys.argv
BASESTR='SIM.'
subs=['adult#001','adult#002','adult#003','adult#004','adult#005','adult#006','adult#007','adult#008','adult#009','adult#010']

if OHIO:
	subs=['559','563','570','575','588','591','540','544','552','567','584','596']
	BASESTR='OHIO.'

if not 'misso' in sys.argv:

	models=['_BEATSONLY','_SIVLAST_BEATSONLY','_SIVFIRST_BEATSONLY','-SB','-nogate','-norestrict','_BEATSONLY-DIRECT','-nodirb','']
	modnames=['Encoder/Decoder','SIV Fine-tune','SIV Initialize','High Capacity','No Gating','No Restriction','Only Dec. SIV Input','No Dec. SIV Input','Full Model']
	outcomes=['.FINAL_RMSE_out','.FINALMAEout']
	outss=outcomes


	#for T test
	blT=[]
	oT=[]

	outs=[]
	outb=[]

	#BLs subs by rmse vs mae by error/use/low/high
	BL=np.zeros((4,len(subs),2,4))
	for mm in range(len(models)):

		m=models[mm]
		out='\\quad  ' + modnames[mm]

		
			
		for oo in range(len(outcomes)):
			o=outcomes[oo]
			out+=' & '


			errs=[]
			lqs=[]
			hqs=[]
			uses=[]


			for s in range(len(subs)):
				ss=BASESTR+subs[s]
				name=ss+m
				nameno=ss+'_BEATSONLY_NOINSCARBS'

				for f in os.listdir(name):
					if f.endswith(o):
						errs.append(float( f[:7]))
						if mm==3:
							blT.append(float( f[:7]))
						if mm==8:
							oT.append(float( f[:7]))
				for f in os.listdir(nameno):
					if f.endswith(o):
						uses.append(float(f[:7])-errs[-1])




				if OHIO:
					atempo=joblib.load('/data3/interns/postohio/allohiodata/'+subs[s]+'.train.pkl')
					gtempo=np.asarray(atempo['glucose'])#/400
					SCALEVAL=np.nanmax(gtempo)

				else:
					atempo=np.array(joblib.load('/data3/interns/postohio/NOSNACKsimsubs_X5/'+subs[s]+'_1.pkl')[:10])
					SCALEVAL=np.max(atempo[:,:,0])



				allpred=[]
				alltarg=[]
				targs=joblib.load(name+'/model0/targs.pkl')
				preds=joblib.load(name+'/model0/preds.pkl')
				pred=np.zeros(0)
				targ=np.zeros(0)
				for t in range(len(targs)):
					targ=np.concatenate((targ,targs[t][:,-1]*SCALEVAL))
					pred=np.concatenate((pred,preds[t][:,-1]*SCALEVAL))


				rmses=[]
				BOOTS=1000
				if 'quick' in sys.argv:
					BOOTS=10
				for t in range(BOOTS):
					rmse=[]
					inds=[]
					for tt in range(len(pred)):
						inds.append(np.random.randint(len(pred)))
					inds=np.array(inds)
					if oo==0:	
						rmses.append( np.sqrt( np.mean( ( pred[inds]-targ[inds] )**2 ) ) )
					else:
						rmses.append( np.mean( np.abs( pred[inds]-targ[inds] ) ) ) 
				rmses=np.sort(rmses)
				lqs.append(rmses[int(.025*BOOTS)])
				hqs.append(rmses[int(.975*BOOTS)])

			out+=str(round(  np.mean(errs)*100)/100)
			out+=',['
			
			outtemp=str( round(np.mean(lqs)*100)/100 )
			out+=outtemp
			out+=','
			outtemp=str( round(np.mean(hqs)*100)/100 )
			out+=outtemp
			out+=']'
			out+=' ('+str( round(np.mean(uses)*100)/100 )
			out+=')'


				

		out+='\\\\'
		print(out)
		outs.append(out)

	print('')
	print('')
	print('t-test.')
	print(scipy.stats.ttest_rel(blT,oT))
	print('')
	print('')











	####THIS SECTION MAKES THE SUBJECT COMPARISON PLOTS#######
if not 'misso' in sys.argv or 'sub2' in sys.argv:

	import joblib
	import numpy as np
	import matplotlib.pyplot as plt


	ALLBL=False
	ALSODIRECTBL='ad' in sys.argv
	subs=['adult#001','adult#002','adult#003','adult#004','adult#005','adult#006','adult#007','adult#008','adult#009','adult#010']

	BASESTR='SIM.'
	checkvals=['']

	if OHIO:
		subs=['559','563','570','575','588','591','540','544','552','567','584','596']
		BASESTR='OHIO.'


	uses=[]
	improve=[]
	bls=[]
	boots=[]
	for ss in range(len(subs)):
		s=subs[ss]
		blfile=BASESTR+s+'_BEATSONLY/'
		filen=BASESTR+s+'_BEATSONLY_NOINSCARBS/'
		file=BASESTR+s+'/'
		use=0
		for i in os.listdir(blfile):
			if i.endswith('_out'):
				err=float(i[:8])

		for i in os.listdir(filen):
				if i.endswith('_out'):
					use=float(i[:8])
		for i in os.listdir(file):
				if i.endswith('_out'):
					ours=float(i[:8])


		x=joblib.load(file+'model0/xs.pkl')[0]
		if OHIO:
			atempo=joblib.load('/data3/interns/postohio/allohiodata/'+s+'.train.pkl')
			gtempo=np.asarray(atempo['glucose'])#/400
			SCALEVAL=np.nanmax(gtempo)
		else:
			atempo=np.array(joblib.load('/data3/interns/postohio/NOSNACKsimsubs_X5/'+s+'_1.pkl')[:10])
			SCALEVAL=np.max(atempo[:,:,0])


		bls.append(err)
		uses.append(use-err)


		improve.append(err-ours)


		
		pred=joblib.load(file+'/model0/preds.pkl')[0][:,-1]*SCALEVAL
		predbl=joblib.load(blfile+'/model0/preds.pkl')[0][:,-1]*SCALEVAL
		targ=joblib.load(file+'/model0/targs.pkl')[0][:,-1]*SCALEVAL
		rmses=[]

		for t in range(100):
			inds=[]
			for tt in range(pred.shape[0]):
				inds.append(np.random.randint(pred.shape[0]))
			inds=np.array(inds)
			rmses.append(np.sqrt( np.mean( ( predbl[inds]-targ[inds] )**2 ) )  -   np.sqrt( np.mean( ( pred[inds]-targ[inds] )**2 ) )  )


		boots.append(np.std(rmses))




	uses=np.array(uses)
	improve=np.array(improve)




	plt.figure(figsize=(12,6)) 
	plt.subplot(1,2,1)
	plt.errorbar(uses,improve,yerr=np.array(boots),marker='o',linewidth=0,elinewidth=1)
	plt.plot([np.min(uses),np.max(uses)],[0,0],'k--')
	print('USE COR: ')
	print(scipy.stats.pearsonr(uses,improve))
	plt.title('(a)',fontsize=22)
	plt.xlabel('Enc/Dec SIV Useage',fontsize=22)
	plt.ylabel('Improvement Over Baseline',fontsize=22)
	plt.xticks(fontsize=12 )
	plt.yticks(fontsize=12 )



	plt.subplot(1,2,2)
	plt.errorbar(bls,improve,yerr=np.array(boots),marker='o',linewidth=0,elinewidth=1)
	plt.plot([np.nanmin(bls),np.nanmax(bls)],[0,0],'k--')
	plt.xlabel('Enc/Dec Error',fontsize=18)
	print('err COR: ')
	plt.title('(b)',fontsize=22)
	print(scipy.stats.pearsonr(bls,improve))
	plt.xticks(fontsize=12 )
	plt.yticks(fontsize=12 )

	#f.set_figheight(1)

	plt.savefig('OUT-PLOT'+BASESTR+'.DOUB.png')
	plt.clf()









####THIS SECTION MAKES THE MISSINGNESS PLOT#######




ann=[0,10,20,30,40,50]
ans=['0','.1','.2','.3','.4','.5']
seeds=['.seed1','.seed2','.seed3','.seed4','.seed5']
subs=range(10)#[9]]
e=np.zeros((2,2,6,len(subs)*len(seeds)))#bl vs ours,mean vs error,misspoint, sub
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['FreeSans', 'Arial', 'Helvetica']

for n in ['miss','noise']:
	subi=0
	for s in subs:
		substr='0'+str(s+1)
		if s==9:
			substr=str(s+1)
		
		
		an=0
		for f in ans:
			for g in [0,1]:
				

				for se in range(len(seeds)):
					basestr='SIM.adult#0'+substr+'.'+n
					if g==0:
						ff=basestr+f+seeds[se]+'_BEATSONLY'
					elif g==1:
						ff=basestr+f+seeds[se]
					targs=joblib.load(ff+'/model0/targs.pkl')
					preds=joblib.load(ff+'/model0/preds.pkl')
					predsbl=joblib.load(basestr+f+'_BEATSONLY'+'/model0/preds.pkl')

					atempo=np.array(joblib.load('/data3/interns/postohio/NOSNACKsimsubs_X5BIGDELAY/adult#0'+substr+'_1.pkl')[:10])
					SCALEVAL=np.max(atempo[:,:,0])

					pred=preds[0][:,-1]*SCALEVAL
					predbl=predsbl[0][:,-1]*SCALEVAL
					targ=targs[0][:,-1]*SCALEVAL

					e[g,0,an,subi+se*len(subs)]=np.sqrt( np.mean( ( pred-targ )**2 ) )
					
					rmses=[]
					for t in range(100):
						inds=[]
						for tt in range(pred.shape[0]):
							inds.append(np.random.randint(pred.shape[0]))
						inds=np.array(inds)
						rmses.append( np.sqrt( np.mean( ( pred[inds]-targ[inds] )**2 ) )  )
					e[g,1,an,subi+se*len(subs)]=np.std(rmses)
			an+=1
		subi+=1

	#manually inserted no SIV value
	#plt.plot([np.min(an),np.max(an)],[28.40864196664824,28.40864196664824],'k:')
	if n=='miss':
		plt.figure(figsize=(12,6)) 
		plt.subplot(1,2,1)
		plt.title('(a)',fontsize=22)
		plt.ylabel('Prediction rMSE',fontsize=13)
	else:

		plt.subplot(1,2,2)
		plt.title('(b)',fontsize=22)
	temp=plt.errorbar(ann,np.mean(e[0,0],1),np.mean(e[0,1],1))
	
	temp=plt.errorbar(ann,np.mean(e[1,0],1),np.mean(e[1,1],1),linestyle='--')
	temp[-1][0].set_linestyle('--')

	plt.legend(['Baseline','Proposed',],fontsize=22)
	plt.xticks(fontsize=12 )
	plt.yticks(fontsize=12 )

	if n=='miss':
		plt.xlabel('% of Carbohydrate Values Missing',fontsize=22)
		#plt.savefig('OUT-carbmissplot.png')
	else:
		plt.xlabel('% Magnitude of Noise',fontsize=22)

plt.savefig('OUT-missnoise.png')

plt.clf()




	




