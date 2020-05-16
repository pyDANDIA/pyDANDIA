### RoboNoise is a implementation of Bramich and Freudling 2012 red noise estimation for a photometric dataset.
### It is implemented to fit continuous quantities, i.e mi = M_j+f(xi), in place of bin offset as it is in the original paper.
### The idea is similar, just some maths changed, especially on the B and D matrices (see Bramich and Freudling 2012)

# Basic python package import
import numpy as np
import collections
import time
### Definition of the RedNoise solver class

class RedNoiseSolver(object):


	def __init__(self, data, dictionary) :
		""" This function set how the class should be call. It is madatory that the class got a dataset parameter, data here, and a dictionnary (in a python meaning)
		    which details wich columns of the dataset corresponds to which quantity. Example, with a dataset :
	    
		    data = [stars,time,frames,magnitude,error_magnitude,airmass] 
		    
		    should have a dictionnary like
	
		    dictionary = {'stars':0,'time':1,'frames':2,'mag':3,'err_mag':4,'airmass':5}
		    Note that for this example, we would only be able to fit the 'airmass' effect.
		
		    Please look to the function for the exact name expected for quantities"""

		self.data = data
		self.dictionary = dictionary
		self.model_quantities = collections.namedtuple('Models',[])
		self.ref_star =[]	

	

	def  construct_continuous_matrices(self,choices):
		""" Construc the matrices as defined in Bramich and Freudling 2012. WARNING : the B and D submatrices are different because the quantities here
		    are continuous i.e :
		    Q_i,j, k : -i indicate the quantity define in the find_model_quantities dataset. n different quantities in total.
		   	       -j indicate the j star. S different stars in total 
			       -k indicate the k exposure (i.e frame). E different exposure in total

		    w_j,k : the inverse magnitude error of the m_j,k magnitude point. 

		    B = [sum_k Q_1,1,k*w_1,k^2 ; sum_k Q_2,1,k*w_1,k^2 ; .....; sum_k Q_n,1,k*w_1,k^2]
		        [sum_k Q_1,2,k*w_2,k^2 ; sum_k Q_2,2,k*w_2,k^2 ; .....; sum_k Q_n,2,k*w_2,k^2]
		        [          .           ;           .           ;   .  ;          .           ]
		        [          .           ;           .           ;   .  ;          .           ]
		        [          .           ;           .           ;   .  ;          .           ]
		        [sum_k Q_1,S,k*w_S,k^2 ; sum_k Q_2,S,k*w_S,k^2 ; .....; sum_k Q_n,S,k*w_S,k^2]	

		    So the B matrice is (S,n) shape.

		    sum_k,j indicates a double sum on both terms.

		    D = [sum_k,j Q_1,j,k*Q_1,j,k*w_j,k^2 ; sum_k,j Q_1,j,k*Q_2,j,k*w_j,k^2 ; .....; sum_k,j Q_1,j,k*Q_n,j,k*w_j,k^2 ]
			[sum_k,j Q_2,j,k*Q_1,j,k*w_j,k^2 ; sum_k,j Q_2,j,k*Q_2,j,k*w_j,k^2 ; .....; sum_k,j Q_2,j,k*Q_n,j,k*w_j,k^2 ]
                        [sum_k,j Q_3,j,k*Q_1,j,k*w_j,k^2 ; sum_k,j Q_3,j,k*Q_2,j,k*w_j,k^2 ; .....; sum_k,j Q_3,j,k*Q_n,j,k*w_j,k^2 ]
			[              .                 ;                .                ; .....;               .                 ]
			[              .                 ;                .                ; .....;               .                 ]
			[              .                 ;                .                ; .....;               .                 ]
			[sum_k,j Q_n,j,k*Q_1,j,k*w_j,k^2 ; sum_k,j Q_n,j,k*Q_2,j,k*w_j,k^2 ; .....; sum_k,j Q_n,j,k*Q_n,j,k*w_j,k^2 ]

		    So the D matrice is (n,n) shape. Note that it is a symmetric, as in Bramich and Freudling 20121, but not diagonal due to derivative.
		
		    
		"""

                # Determine the star list in the dataset
		stars, indexes_stars, count_stars = np.unique(self.data[:,self.dictionary['stars']], return_index=True, return_counts=True)
		number_of_stars = len(stars)

		# Determine the frame list in the dataset
		frames = np.unique(self.data[:,self.dictionary['frames']])
		number_of_frames = len(frames)

		# create quantity to fit as selected in choice. If you choose 'airmass' and 'seeing', then quantities will be quantities = [Q_'airmass',Q_'seeing']. See documentation of the function for details on quantities.

		quantities = self.find_model_quantities(choices[0])
				
		# stack the quantities column by columne : quantities =[Q1,Q2,Q3....Qn]

		for i in choices[1:] :
			quantities = np.c_[quantities,self.find_model_quantities(i)]
		
		# normalise quantities with errorbar
		if quantities.ndim == 1:

			quantities = quantities/self.data[:,self.dictionary['err_mag']].astype(float)
		else:

			quantities = quantities/self.data[:,self.dictionary['err_mag']].astype(float)[:,None]	

                # MAKE CHECK HERE THAT THERE ARE AT LEAST TWO STARS - MORE ROBUST CODE - AND REPORT AN ERROR MESSAGE IF NOT. TODO!

		# Construct the A, B sub matrices, see Bramich and Freudling 2012, and the v1 vector.

		start = time.time()	
		A_diagonal = []
		v1 = []
		B=[]
		

		for i in xrange(len(stars)) :
			

			index = np.arange(indexes_stars[i],indexes_stars[i]+count_stars[i]).tolist()			

			A_diagonal.append(sum(1/self.data[index,self.dictionary['err_mag']].astype(float)**2))
			v1.append(sum(self.data[index,self.dictionary['mag']].astype(float)/self.data[index,self.dictionary['err_mag']].astype(float)**2))
			line=[]
			if quantities.ndim==1:
					
				line+=[np.sum(quantities[index]*1/self.data[index,self.dictionary['err_mag']].astype(float))]					
			
			else:

				line+=np.sum(quantities[index].T*1/self.data[index,self.dictionary['err_mag']].astype(float),axis=1).tolist()

				
			B.append(line)

		
		## Old way		

		#for i in stars :
			
		#	index = np.where(self.data[:,self.dictionary['stars']]==i)[0]
		#	A_diagonal.append(sum(1/self.data[index,self.dictionary['err_mag']].astype(float)**2))
		#	v1.append(sum(self.data[index,self.dictionary['mag']].astype(float)/self.data[index,self.dictionary['err_mag']].astype(float)**2))
		#	line=[]
		#	if quantities.ndim==1:
					
		#		line+=[np.sum(quantities[index]*1/self.data[index,self.dictionary['err_mag']].astype(float)**2)]					
			
		#	else:

		#		line+=np.sum(quantities[index].T*1/self.data[index,self.dictionary['err_mag']].astype(float)**2,axis=1).tolist()

				
		#	B.append(line)

		
		B=np.array(B)
		#import pdb; pdb.set_trace()	
		print 'Matrices A,B and v1 construct in',time.time()-start,'s'
		
		
		# Construct the D matrix and v2 vector
		
		
		start = time.time()


		v2=[]
		if quantities.ndim==1:
			n_dim = 1
			D = np.zeros((1,1))
			quantities_i=quantities
			quantities_j=quantities
			D[0,0] = np.sum(quantities_i*quantities_j)
			v2.append(np.sum(quantities_i*self.data[:,self.dictionary['mag']].astype(float)/self.data[:,self.dictionary['err_mag']].astype(float)))
			v2=np.array(v2)
		else:
			n_dim =quantities.shape[1]
			D = np.zeros((n_dim,n_dim))

			for i in xrange(n_dim)  :
				
				quantities_i=quantities[:,i]
				
				for j in np.arange(i,n_dim) :
			
					quantities_j=quantities[:,j]
				
					
					if choices[0]=='frames' :
						#import pdb; pdb.set_trace()
						if i==j :

							somme =  np.sum(quantities_i*quantities_j)
						else :
					
							somme  = 0.0
					else :
						somme =  np.sum(quantities_i*quantities_j)
					D[i,j] = somme
					D[j,i] = somme
					
			
				v2.append(np.sum(quantities_i*self.data[:,self.dictionary['mag']].astype(float)/self.data[:,self.dictionary['err_mag']].astype(float))) 
		
		
			v2=np.array(v2)

		print 'Matrices D and v2 construct in',time.time()-start,'s'
		self.A_diagonal=np.array(A_diagonal)
		self.B=B
		self.D=D
		self.v1=v1
		self.v2=v2
		print 'Matrices construction : OK, we are now close to the solution'

	def find_model_quantities(self,choice) :
		""" Return for each choice the quantities defin in each subfunction"""

		if choice == 'seeing' :
			return self.model_seeing() 
		
		if choice == 'airmass' :
			return self.model_airmass() 
		if choice == 'phot_scale_factor' :
			return self.model_phot_scale_factor() 	
		if choice == 'CCD' :
			return self.model_CCD_positions() 
		if choice == 'exposure' :
			return self.model_exposure_time() 
		if choice == 'background' :
			return self.model_background() 

		if choice == 'frames' :
			return self.model_frames()

	# definitions of continuous quantities functions. Mainly linear functions for a start.
	def model_frames(self) :
		
		frames = np.unique(self.data[:,self.dictionary['frames']])
		stars = np.unique(self.data[:,self.dictionary['stars']])
			
		count = 0
		for i in frames :
			Quantity = np.zeros((len(self.data)))
			index = np.where(self.data[:,self.dictionary['frames']]==i)[0]
			Quantity[index] = 1.0
			if count == 0 :
			
				quantity = Quantity
				count += 1
			else :
			
				quantity = np.c_[quantity,Quantity]		
			
					
		return quantity

	def model_airmass(self) :
		# f(x) = a*x --> Q_'airmass' = [airmass]
		quantity = self.quantities.airmass
		#quantity = np.c_[quantity,self.quantities.airmass**2]
		#quantity = np.c_[quantity,self.quantities.airmass**3]
		return quantity


	def model_CCD_positions(self) :
		#import pdb; pdb.set_trace()	
		#f(x) = a_i*x**i*y**j  --> Q_'CCD' =[CCD_Y^1,CCD_Y^2,....CCD_Y^j*CCD_X^i,....,CCD_X^i] 
		
		degree = self.CCD_fit_degree
		offset_CCD = np.array(len(self.quantities.CCD_X)*[1.0])
		for i in range(degree+1) :
			for j in range(degree+1) :
			
				if (i+j>degree) | (i+j==0) :	
					pass
				else :

					offset_CCD = np.c_[offset_CCD, ((self.quantities.CCD_X))**(i)*((self.quantities.CCD_Y))**(j)]

		return offset_CCD[:,0:]
		
	
	def model_exposure_time(self) :
		# f(x) = a*x --> Q_'exposure_time' = [exposure_time]
		offset_exptime =self.quantities.exposure
		#offset_exptime =np.c_[offset_exptime ,self.quantities.exposure**2]
		return offset_exptime

	def model_seeing(self) :
		# f(x) = a*x  --> Q_'seeing' = [seeing]
		offset_seeing = self.quantities.seeing
		return offset_seeing
	def model_background(self) :
		# f(x) = a*x --> Q_'background' = [background]
		offset_background = self.quantities.background
		return offset_background

	def model_phot_scale_factor(self) :
		# f(x) = a*x  --> Q_'phot_scale_factor' = [phot_scale_factor]
		offset_phot_scale_factor =self.quantities.phot_scale_factor
		return offset_phot_scale_factor
	 
	def clean_bad_data(self) :
		

		# Delete all the measurement for the star j if  : error_mag is -1.0 or >10; or if the variation of the star is to big (i.e std(mag)>1)
		 

		#self.data = self.data[self.data[:,self.dictionary['stars']].argsort(),]

                mask = ((self.data[:,self.dictionary['err_mag']].astype(float)==-1.0) | (self.data[:,self.dictionary['err_mag']].astype(float)>10))
		good_measurements = np.where(mask==False)[0]

		good_data = self.data[good_measurements]
		
		stars, indexes, count =  np.unique(good_data[:,self.dictionary['stars']], return_index=True, return_counts = True)
		#stars, indexes, count =  np.unique(self.data[good_measurements], return_index=True, return_counts = True)
		index = []
		max_number_of_measurements = max(count)
		for i in xrange(len(count)) :
			
			if count[i] == max_number_of_measurements :
				index_star = np.arange(indexes[i],indexes[i]+count[i]).tolist()
				if np.std(good_data[index_star,self.dictionary['mag']].astype(float))<1.0:
				#if np.std(self.data[good_measurements[index_star],self.dictionary['mag']].astype(float))<1.0:	
					index += index_star
		
		## Old way to do it, much much more slower
	
		#stars = np.unique(self.data[:,self.dictionary['stars']])
		#mask = ((self.data[:,self.dictionary['err_mag']].astype(float)==-1.0) | (self.data[:,self.dictionary['err_mag']].astype(float)>10))
		#bad_measurements = np.where(mask==True)[0]
		#bad_stars = np.unique(self.data[bad_measurements,self.dictionary['stars']])	
		#good_stars = np.setdiff1d(stars,bad_stars)
		#index3 = []
		#for i in good_stars :
		#	start = time.time()
		#	index2 = np.where(self.data[:,self.dictionary['stars']]==i)[0]
			
		#	print time.time()-start
		#	if np.std(self.data[index2,self.dictionary['mag']].astype(float))<1.0 :
		#		index3 += index2.tolist() 
			
		#imporpdb; pdb.set_trace()	
		self.data = good_data[index]
		#self.data = self.data[good_measurements[index]]			
		print 'Bad data clean : OK'

	def clean_bad_stars(self,choices) :

		#Clean stars that you do not want, for example the microlensing target. 

		stars = np.unique(self.data[:,self.dictionary['stars']])
		index = np.arange(0,len(self.data))
		index2 = []		
		
		for i in choices :

			index2 += np.where(self.data[:,self.dictionary['stars']]==i)[0].tolist()	
		
		index = np.delete(index,index2)			
			
		
		self.data = self.data[index]
		print 'Bad stars clean : OK '
			
	def clean_magnitude_data(self,threshold) :
		
		#Clean stars fainter than the threshold. 




		
		
                mask =(self.data[:,self.dictionary['mag']].astype(float)<22)
		
		
		good_data = self.data[mask]
		
		stars, indexes, count =  np.unique(good_data[:,self.dictionary['stars']], return_index=True, return_counts = True)
		index = []
		max_number_of_measurements = max(count)
		for i in xrange(len(count)) :
			
			if count[i] == max_number_of_measurements :
				index_star = np.arange(indexes[i],indexes[i]+count[i]).tolist()					
				index += index_star
		
		self.data = good_data[index]
		## Old way to do it, much much more slower


		#stars = np.unique(self.data[:,self.dictionary['stars']])
		#index = np.arange(0,len(self.data)+1)
		#index3 = []
		
		
		#for i in stars :

			#index2 = np.where(self.data[:,self.dictionary['stars']]==i)[0]			
			#if max(self.data[index2,self.dictionary['mag']].astype(float))>threshold :
			#	pass
			#else :
		
			#	index3 = index3+index2.tolist()
	


		#self.data = self.data[index]			
		print 'Stars magnitude clean : OK'



	def define_continuous_quantities(self,choices):
		
		# Define the quantities for the solver. If this is in choices, the code read the column thanks to the dictionary. If not, just set it to zero.

	
		#interest=['airmass','CCD_X','CCD_Y','exposure','background','seeing','time','phot_scale_factor']
		quantities=collections.namedtuple('Quantities',choices)
		
		for i in choices :
				
			if i !='frames' :
			
				setattr(quantities,i,self.data[:,self.dictionary[i]].astype(float))
			
			

				
		
		
		self.quantities = quantities


	def solve(self) :
		
		# Solve the least-square problem, as defined in BRamich and Freudling 2012.
		
		A_diagonal=self.A_diagonal
		B=self.B
		C=self.B.T		
		D=self.D
		v1=self.v1
		
		v2=self.v2

		####Set the first star magnitude to the median, to scale the problem. Comment this to avoid that.
		#### !Warning on the the term self.x1 below, which is computed with the good quantities (without this scaling)
		
		first = np.unique(self.data[:,self.dictionary['stars']])[0]
		first_star = np.where(self.data[:,self.dictionary['stars']] == first)[0]
		
		V1 = v1[1:]
		V2 = v2 - np.dot(C[:,0],np.median(self.data[first_star,self.dictionary['mag']].astype(float)))
		A_Diagonal = A_diagonal[1:]
		BB = B[1:,:]
		CC = C[:,1:]

		## Uncomment this if you dont want to scale the pb (probably a bad idea)
		#A_Diagonal = A_diagonal
		#BB = B
		#CC = C
		#V1 = v1
		#V2 = v2
		

		Invert_A_Diagonal=1/A_Diagonal	
		
		
		term1=Invert_A_Diagonal[:,None]*BB	
		term2=np.dot(CC,term1)
		term3=D-term2
		term4=Invert_A_Diagonal*V1
		term5=np.dot(CC,term4)
		term6 = V2-term5
			
		# inverting matrix, not really efficient 
		#Invert = np.linalg.inv(term3)
		#x2=np.dot(Invert,term6)


		#x2=np.linalg.solve(term3,term6)# solver, not the best

		#leastsq solver
		x2=np.linalg.lstsq(term3,term6)[0]



		self.x2 = x2
		self.x1=1/A_diagonal*v1-1/A_diagonal*np.dot(B,x2)
		
###########################################################################################################################################################################
### ALL THE FOLLOWING CAN, AND SHOULD, BE IGNORED. THIS COMES FROM ORIGINAL BRAMICH AND FREUDLING 2012 IMPLEMENTATION. WE KEEP IT FOR HISTORIC AND DEVELOPMENT PURPOSE ####
###########################################################################################################################################################################

	def define_bins(self,choice,size=0.0):
		#import pdb; pdb.set_trace()
		if size ==0.0 :
			
			size = len(np.unique(self.data[:,self.dictionary[choice[0]]]))
			self.bins = np.unique(self.data[:,self.dictionary[choice[0]]]).astype(float)
			self.bins = np.append([self.bins],[2*self.bins[-1]])

		else :

			#define bins with equally spaced, self.bins are bins limits
			histogram = np.histogram(self.data[:,choice].astype(float),size)
			self.bins = histogram[1]

	
	def compute_quantities(self,choices):

		quantities=collections.namedtuple('Bins',choices)
		
		for i in choices :
			#import pdb; pdb.set_trace()
			
                        # Determine the unique set of epochs (or images) which contain the stars from whic
                        # the measurements are derived
			if i=='time' :	
				quantities.time = np.unique(self.data[:,self.dictionary[i]].astype(float))

                        # NOT SURE WHY YOU DO THIS FOR A POTENTIALLY CONTINUOUS QUANTITY - ADMITTEDLY USUALLY A SINGLE AIRMASS
                        # VALUE IS ASSOCIATED WITH EACH IMAGE, BUT IN REALITY AIRMASS VARIES ACROSS THE IMAGE AREA. SO FOR
                        # THE CODE TO BE GENERAL AIRMASS SHOULD BE TREATED AS A CONTINUOUS QUANTITY, AND THEREFORE A VECTOR
                        # OF UNIQUE VALUES IS NOT A GOOD WAY TO GO
			if i=='airmass' :
				quantities.airmass = np.unique(self.data[:,self.dictionary[i]].astype(float))	

                        # Determine the unqiue set of exposure times used
			if i=='exposure' :	
				quantities.exposure = np.unique(self.data[:,self.dictionary[i]].astype(float))			

		self.quantities = quantities

	# Construct the matrices in the linear least squares problem
	def construct_matrices(self,choices):
             
			#
			self.compute_quantities(choices)


                        # Determine the star list
			stars = np.unique(self.data[:,self.dictionary['stars']])
			number_of_stars = len(stars)

                        # MAKE CHECK HERE THAT THERE ARE AT LEAST TWO STARS - MORE ROBUST CODE - AND REPORT AN ERROR MESSAGE IF NOT

                        # I THINK THIS IS UNNECESSARY AND UNWEILDY - THE PROBLEM POTENTIALLY HAS MANY MANY STARS. ONLY THE DIAGONAL OF THIS MATRIX IS NON-ZERO.
			# NO NEED TO STORE THE WHOLE MATRIX
			A=np.zeros((number_of_stars,number_of_stars))

			A_diagonal=[sum(1/self.data[np.where(self.data[:,self.dictionary['stars']]==i)[0],self.dictionary['err_mag']].astype(float)**2) for i in stars]
			np.fill_diagonal(A,A_diagonal)
			
			# A_DIAGONAL AND V1 CAN BE CALCULATED IN THE SAME LOOP TO AVOID REPEATING COSTLY OPERATIONS MORE THAN NECESSARY e.g. np.where(self.data[:,self.dictionary['stars']]==i)
			v1=[sum(self.data[np.where(self.data[:,self.dictionary['stars']]==i)[0],self.dictionary['mag']].astype(float)/self.data[np.where(self.data[:,self.dictionary['stars']]==i)[0],self.dictionary['err_mag']].astype(float)**2) for i in stars]

			# Construct the B sub-matrix
			B=[]
			for i in xrange(number_of_stars):
				line=[]
				index = np.where(self.data[:,self.dictionary['stars']]==stars[i])[0]
				for j in self.quantities._fields:
					matches = getattr(self.quantities, j)
					#import pdb; pdb.set_trace()
					for k in matches :
						#import pdb; pdb.set_trace()
						index2=np.where(self.data[index,self.dictionary[j]].astype(float)==k)[0]	
						line+=[np.sum(1/self.data[index[index2],self.dictionary['err_mag']].astype(float)**2)]
				B.append(line)
				print i
			B=np.array(B)
			
			# Construct the D matrix and v2 vector
			D=np.zeros((B.shape[1],B.shape[1]))
			v2=np.zeros(B.shape[1])
			count=0
			for i in self.quantities._fields :
				matches = getattr(self.quantities,i)
				for k in matches :
					index=np.where(self.data[:,self.dictionary[i]].astype(float)==k)[0]
					D[count,count]=sum(1/self.data[index,self.dictionary['err_mag']].astype(float)**2)
					v2[count] = sum(self.data[index,self.dictionary['mag']].astype(float)/self.data[index,self.dictionary['err_mag']].astype(float)**2)
					count+=1
	
			
			self.A=A
			self.B=B
			self.D=D
			self.v1=v1
			self.v2=v2	
			#import pdb; pdb.set_trace()

	
		
