import glob
import numpy as np
import nibabel as nib
import os
import pickle

# min no. of annoted pixel to keep
thres=0

# get all the scans and annots
files=glob.glob('./data/**/*.nii',recursive=True)

# organise the files patientwise
patientsFiles={file.split('/')[2]:[] for file in files}
for file in files:
    patientsFiles[file.split('/')[2]].append(file) 
    
# for every patient
    ## get nonzero thresholded annotslices in each view
    ## save the respective slices ineach modality

path='./dataPreprocessed/'
views=['sagital/','frontal/','coronal/']
for patient in patientsFiles.keys():
    annotPath=sorted(patientsFiles[str(patient)])[-1]
    annot=nib.load(annotPath).get_fdata()
    
    # non zero slices in each axis 
    locs_saggital, locs_frontal, locs_coronal= np.where(annot==1)   
    
    # pixel count per slice
    pix_count_saggital = np.bincount(np.sort(locs_saggital))
    pix_count_frontal = np.bincount(np.sort(locs_frontal)) 
    pix_count_coronal = np.bincount(np.sort(locs_coronal))
    
    # thresholded location
    loc_condition_saggital = np.squeeze(np.asarray(np.where(pix_count_saggital >= thres)))
    loc_condition_frontal = np.squeeze(np.asarray(np.where(pix_count_frontal >= thres)))
    loc_condition_coronal = np.squeeze(np.asarray(np.where(pix_count_coronal >= thres)))
    
    # create folder structure to save data
    pathS=[[path+view+patient,os.makedirs(path+view+patient,exist_ok=True)] for view in views]

    
    # save data
    for scan in sorted(patientsFiles[str(patient)])[:-1]:
        heading=scan.split('.')[-3]
        scan=nib.load(scan).get_fdata()
        
        #normalise scan [0,1]
        scan-=scan.min()
        scan/=scan.max()
        
        scansSelectedSlice=[scan[loc_condition_saggital,:,:], scan[:,loc_condition_frontal,:], scan[:,:,loc_condition_coronal]]
        
        for i, scanView in enumerate(scansSelectedSlice):
            with open(pathS[i][0]+'/'+heading, 'wb') as f:
                pickle.dump(scanView, f)
                f.close()
                
    # save annot
    annotsSelectedSlice=[annot[loc_condition_saggital,:,:], annot[:,loc_condition_frontal,:], annot[:,:,loc_condition_coronal]]
        
    for i, scanView in enumerate(annotsSelectedSlice):
        with open(pathS[i][0]+'/annot', 'wb') as f:
            pickle.dump(scanView, f)
            f.close()           
                
# Consolidate the dataset(ignore patient info)
    ## for every view
        ### collect all the patient scan for a modality
        ### collect all the annots
        
modalities=['MR_DWI','MR_Flair','MR_T1','MR_T2','annot']
for view in views:
    for modality in modalities:
        
        # paths to all scans in a modality
        allScans=glob.glob(path+view+'/**/*'+modality)
            
        scanImage=np.load(allScans[0])
        if len(scanImage.shape)<3:
            scanImage=np.expand_dims(scanImage, views.index(view))
            
        # merge all scans
        for scan in allScans[1:]:
            img=np.load(scan)
            if len(img.shape)<3:
                img=np.expand_dims(img, views.index(view))
                
            if views.index(view)!=2:
                scanImage=np.concatenate([scanImage[:,:,:153],img[:,:,:153]],views.index(view))
            elif views.index(view)==2:
                scanImage=np.concatenate([scanImage,img],views.index(view))
 
        # save scan
        with open(path+view+'/'+modality, 'wb') as f:
            pickle.dump(scanImage, f)
            f.close()
