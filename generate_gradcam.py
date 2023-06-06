from torch.utils.data import DataLoader
# from inception_resnet2_7 import InceptionResNetV2
# from inception_resnet_v2 import InceptionResNetV2
import torch
from skimage.transform import resize
import SimpleITK as sitk
import time
import numpy as np
import os
import torch.nn as nn
from grad_cam_github.grad_cam import GuidedBackPropagation, GradCAM
import argparse
# from resnet3d_attention import generate_resnet3d
import zipfile
from shutil import rmtree

def retrieve_file_paths(dirName):
    # setup file paths variable
    filePaths = []

    # Read all directory, subdirectories and file lists
    for root, directories, files in os.walk(dirName):
        for filename in files:
            # Create the full filepath by using os module.
            filePath = os.path.join(root, filename)
            filePaths.append(filePath)
         
    # return all paths
    return filePaths

def compress(dirName):
    filePaths = retrieve_file_paths(dirName)
    zip_file = zipfile.ZipFile('rams/midl_rebuttal/' + dirName.split('/')[-1]+'.zip', 'w')
    with zip_file:
        for file in filePaths:
            zip_file.write(file)

# def save_dicom(img, pname, mode, task='surv'):
#     if not os.path.exists('rams/{}/{}/{}'.format(task, pname, mode)):
#         os.makedirs('rams/{}/{}/{}'.format(task, pname, mode))
#     if not mode in ['img', 'overlaid']:
#         img = img.astype(float)
#         img -= img.min()
#         img /= img.max()
#         img *= 255.0
#         img = np.uint8(img)
#     new_img = sitk.GetImageFromArray(img)
#     new_img.SetSpacing((1,1,1))
#     writer = sitk.ImageFileWriter()
#     writer.KeepOriginalImageUIDOn()
#     modification_time = time.strftime("%H%M%S")
#     modification_date = time.strftime("%Y%m%d")
#     direction = new_img.GetDirection()
#     series_tag_values = [("0008|0031",modification_time), # Series Time
#                     ("0008|0021",modification_date), # Series Date
#                     ("0008|0008","DERIVED\\SECONDARY"), # Image Type
#                     ("0020|000e", "1.2.826.0.1.3680043.2.1125."+modification_date+".1"+modification_time), # Series Instance UID
#                     ("0020|0037", '\\'.join(map(str, (direction[0], direction[3], direction[6],# Image Orientation (Patient)
#                                                         direction[1],direction[4],direction[7])))),
#                     ("0008|103e", "Created-SimpleITK")] # Series Description
    
#     for i in range(new_img.GetDepth()):
#         image_slice = new_img[:,:,i]
#         # Tags shared by the series.
#         for tag, value in series_tag_values:
#             image_slice.SetMetaData(tag, value)
#         # Slice specific tags.
#         image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d")) # Instance Creation Date
#         image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S")) # Instance Creation Time
#         # Setting the type to CT preserves the slice location.
#         image_slice.SetMetaData("0008|0060", "CT")  # set the type to CT so the thickness is carried over
        
#         # (0020, 0032) image position patient determines the 3D spacing between slices.
#         image_slice.SetMetaData("0020|0032", '\\'.join(map(str,new_img.TransformIndexToPhysicalPoint((0,0,i))))) # Image Position (Patient)
#         image_slice.SetMetaData("0020,0013", str(i)) # Instance Number

#         # Write to the output directory and add the extension dcm, to force writing in DICOM format.
#         writer.SetFileName(os.path.join('rams/{}/{}/{}'.format(task, pname, mode),str(i)+'.dcm'))
#         writer.Execute(image_slice)


def save_dicom(img, pname, folder, type='img'):
    if not os.path.exists('{}/{}/{}'.format(folder,pname,type)):
        os.makedirs('{}/{}/{}'.format(folder,pname,type))
    new_img = sitk.GetImageFromArray(img)
    new_img.SetSpacing((1,1,1))
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    modification_time = time.strftime("%H%M%S")
    modification_date = time.strftime("%Y%m%d")
    direction = new_img.GetDirection()
    series_tag_values = [("0008|0031",modification_time), # Series Time
                    ("0008|0021",modification_date), # Series Date
                    ("0008|0008","DERIVED\\SECONDARY"), # Image Type
                    ("0020|000e", "1.2.826.0.1.3680043.2.1125."+modification_date+".1"+modification_time), # Series Instance UID
                    ("0020|0037", '\\'.join(map(str, (direction[0], direction[3], direction[6],# Image Orientation (Patient)
                                                        direction[1],direction[4],direction[7])))),
                    ("0008|103e", "Created-SimpleITK")] # Series Description
    
    for i in range(new_img.GetDepth()):
        image_slice = new_img[:,:,i]
        # Tags shared by the series.
        for tag, value in series_tag_values:
            image_slice.SetMetaData(tag, value)
        image_slice.SetMetaData("0028|0103", str(1)) # save signed values
        # Slice specific tags.
        image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d")) # Instance Creation Date
        image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S")) # Instance Creation Time
        # Setting the type to CT preserves the slice location.
        image_slice.SetMetaData("0008|0060", "CT")  # set the type to CT so the thickness is carried over
        
        # (0020, 0032) image position patient determines the 3D spacing between slices.
        image_slice.SetMetaData("0020|0032", '\\'.join(map(str,new_img.TransformIndexToPhysicalPoint((0,0,i))))) # Image Position (Patient)
        image_slice.SetMetaData("0020|0013", str(i)) # Instance Number

        # Write to the output directory and add the extension dcm, to force writing in DICOM format.
        writer.SetFileName(os.path.join('{}/{}/{}/'.format(folder,pname,type),str(i)+'.dcm'))
        writer.Execute(image_slice)

def generate_CAMs(model, img):
    pred = model(img)
    model.zero_grad()
    pred.backward()
    hm_vanilla = img.grad.clone()
    hm_vanilla = np.maximum(hm_vanilla, 0)
    hm_vanilla /= torch.max(hm_vanilla)
    hm_vanilla = np.uint8(hm_vanilla*255)
    ## grad cam
    gradients = model.get_gradient()
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3, 4])
    activations = model.get_activations(img).detach()
    for i in range(int(pooled_gradients.size(0))):
        activations[:, i] *= pooled_gradients[i]
    hm_gradCAM = torch.mean(activations, dim=1).squeeze()
    hm_gradCAM = np.maximum(hm_gradCAM, 0)
    hm_gradCAM /= torch.max(hm_gradCAM)
    # hm_gradCAM = resize(hm_gradCAM, list(img.size()[2:]), order=3)
    hm_gradCAM = nn.functional.interpolate(hm_gradCAM, size=[192,256,256], mode='trilinear', align_corners=True)
    hm_gradCAM = np.uint8(hm_gradCAM*255)
    return hm_vanilla.squeeze(), hm_gradCAM, torch.sigmoid(pred)

def guided_gradcam(model, img, args, pname):
    gbp = GuidedBackPropagation(model=model)
    gc = GradCAM(model=model)
    for i in range(img.size(0)):
        _img = img[i][None]
        _ = gbp.forward(_img)
        _ = gc.forward(_img)

        gbp.backward()
        gradients = torch.nn.functional.relu(gbp.generate()).squeeze()

        gc.backward()
        # block4.downsample.0
        regions = gc.generate(target_layer='final_conv').squeeze()
        saliency = torch.mul(gradients,regions).numpy()
        _img = np.uint8(_img.data.squeeze().cpu().numpy()*255)

        save_dicom(_img, pname, 'img')
        save_dicom(saliency, pname, args.mode, args.task)


def gradcam(model, img, clinical, args, pname, save=True, target_layer='final_conv'):
    gcam = GradCAM(model=model)
    for i in range(img.size(0)):
        _img = img[i][None]
        _ = gcam.forward(_img, clinical)

        gcam.backward()
#         for name, layer in model.named_modules():
# ...     if isinstance(layer, torch.nn.Conv2d):
# ...             print(name, layer)
        regions = gcam.generate(target_layer=target_layer).squeeze()

        saliency = regions.numpy()
        saliency[saliency <= np.quantile(saliency[saliency>0],0.9)] = 0
        _img = np.uint8(_img.data.squeeze().cpu().numpy()*255)
    if save:
        saliency = (saliency*255.0).astype(np.uint8)
        save_dicom(_img, pname, 'rams/midl_rebuttal','img')
        save_dicom(saliency, pname, 'rams/midl_rebuttal','gc')
        compress('{}/{}'.format('rams/midl_rebuttal',pname))
        rmtree('{}/{}'.format('rams/midl_rebuttal',pname))
        del gcam
        import gc
        gc.collect()
        
    else:
        return _img, saliency

def guided_backprob(model, img, args, pname):
    gbp = GuidedBackPropagation(model=model)
    for i in range(img.size(0)):
        _img = img[i][None]
        _ = gbp.forward(_img)

        gbp.backward()
        gradients = torch.nn.functional.relu(gbp.generate()).squeeze()

        saliency = gradients.numpy()
        _img = np.uint8(_img.data.squeeze().cpu().numpy()*255)

        save_dicom(_img, pname, 'img', args.task)
        save_dicom(saliency, pname, args.mode, args.task)

def all_cams(model, img, args, pname):
    gc = GradCAM(model=model)
    gbp = GuidedBackPropagation(model=model)
    for i in range(img.size(0)):
        _img = img[i][None]
        _ = gbp.forward(_img)
        _ = gc.forward(_img)

        gc.backward()
        regions = gc.generate(target_layer='final_conv').squeeze()

        gbp.backward()
        gradients = torch.nn.functional.relu(gbp.generate()).squeeze()

        gc_saliency = regions.numpy()
        bp_saliency = gradients.numpy()
        guided_saliency = torch.mul(gradients,regions).numpy()

        _img = np.uint8(_img.data.squeeze().cpu().numpy()*255)
        save_dicom(_img, pname, 'img', args.task)
        save_dicom(gc_saliency, pname, 'gradcam', args.task)
        save_dicom(bp_saliency, pname, 'guided_backprob', args.task)
        save_dicom(guided_saliency, pname, 'guided_gradcam', args.task)

def attention(model, img, args, pname, save_img=True):
    if args.followup:
        img1, img2 = img
        pred, att1, _ = model(img1, True)
        att1 = torch.nn.functional.interpolate(att1, size=(192,256,256), align_corners=True, mode='trilinear')
        att1 = att1.detach().numpy().squeeze()

        pred, att2, _ = model(img2, True)
        att2 = torch.nn.functional.interpolate(att2, size=(192,256,256), align_corners=True, mode='trilinear')
        att2 = att2.detach().numpy().squeeze()

        if save_img:
            img1 = np.uint8(img1.data.squeeze().cpu().numpy()*255)
            img2 = np.uint8(img2.data.squeeze().cpu().numpy()*255)
            save_dicom(img1, pname, 'img1', args.task)
            save_dicom(img2, pname, 'img2', args.task)
        save_dicom(att1, pname, 'attention1', args.task)
        save_dicom(att2, pname, 'attention2', args.task)
    else:
        pred, att, _ = model(img, True)
        att = torch.nn.functional.interpolate(att, size=(192,256,256), align_corners=True, mode='trilinear')
        att = att.detach().numpy().squeeze()
        if save_img:
            img = np.uint8(img.data.squeeze().cpu().numpy()*255)
            save_dicom(img, pname, 'img', args.task)
        save_dicom(att, pname, 'attention', args.task)

    compress('rams/surv/{}'.format(pname))
    rmtree('rams/surv/{}'.format(pname))


def attention_and_gradcam(model,img,args,pname):
    # gradcam(model, img, args, pname, target_layer='layer4.1.conv2')
    attention(model, img, args, pname, True)
    compress('rams/surv/{}'.format(pname))
    rmtree('rams/surv/{}'.format(pname))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', type=str, default='surv')
    parser.add_argument('-fold', type=int, default=1)
    parser.add_argument('-followup', type=int, default=0)
    parser.add_argument('-mode', type=str, default='guided_gradcam')
    # parser.add_argument('-patient', type=int)
    parser.add_argument('-patient', nargs='+', type=int)
    args = parser.parse_args()
    # model = InceptionResNetV2(classes=1, mode='finetune')
    model = generate_resnet3d(classes=1, model_depth=18, in_channels=1, normalization='in', n_attention=2)
    if args.task == 'surv':
#         dl = dataset class
        model.load_state_dict(torch.load('/home/ashahin/codes/survival_analysis/checkpoints/att2/resnet3d_18/resnet3d_18_{}.pt'.format(args.fold), map_location='cpu')['model_state_dict'])
        # model.load_state_dict(torch.load('/home/ashahin/codes/survival_analysis/checkpoints/ipf_no_depth_fix_transplant/inception_resnet2/inception_resnet2_{}.pt'.format(args.fold), map_location=torch.device('cpu'))['model_state_dict'])
        # model.load_state_dict(torch.load('/home/ashahin/codes/survival_analysis/checkpoints/inception_resnet2_lr1e-3_sgd_cosine_augment/inception_resnet2/inception_resnet2_3.pt', map_location=torch.device('cpu'))['model_state_dict'])
        # model.load_state_dict(torch.load('/home/ashahin/codes/survival_analysis/checkpoints/model_fix_inc_res2_aug_1e-3_sgd/inception_resnet2/inception_resnet2_1.pt', map_location=torch.device('cpu'))['model_state_dict'])
    elif args.task == 'gender':
        model.load_state_dict(torch.load('/home/ashahin/codes/survival_analysis/ssl_ckpts/sex_predictor/inception_resnet2/inception_resnet2.pt', map_location=torch.device('cpu'))['model_state_dict'])
        dl = SSLLoader(root_dir='/SAN/medic/IPF', split='val', augment=0, n=1)
    else:
        print("ERROR, choose surv or gender")
    model.eval()
    loader = DataLoader(dl, batch_size=1, shuffle=False)
    
    for sample in loader:
        # label = sample['label']
        if args.followup:
            img1 = sample['img1']
            img2 = sample['img2']
            pname = sample['case'].numpy()[0]
            print(pname)
            if args.mode == 'attention':
                attention(model, (img1,img2), args, pname)
        else:
            img = sample['img']
            try:
                pname = sample['pname'][0][:-3]
            except:
                pname = sample['case'].numpy()[0]
            print(pname)
            # hm_vanilla, hm_gc, pred = generate_CAMs(model, img.requires_grad_())
            if args.mode == 'guided_gradcam':
                guided_gradcam(model, img, args, pname)
            elif args.mode == 'gradcam':
                gradcam(model, img, args, pname)
            elif args.mode == 'guided_backprob':
                guided_backprob(model, img, args, pname)
            elif args.mode == 'all':
                all_cams(model, img, args, pname)
            elif args.mode == 'attention':
                attention(model, img, args, pname)
            elif args.mode == 'AttAndGC':
                attention_and_gradcam(model, img, args, pname)

    # import pandas as pd
    # risk_pred_all, censor_all, survtime_all, case_all = np.array([]), np.array([]), np.array([]), np.array([])
    # df = pd.DataFrame(columns=['pid','death','time','risk_pred'])
    # for i,sample in enumerate(loader):
    #     print(i)
    #     case = sample['case']
    #     censor = sample['event_indicator']
    #     survtime = sample['event_time']
    #     img = sample['img']
    #     with torch.no_grad():
    #         pred = model(img)
    #     risk_pred_all = np.concatenate((risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))   # Logging Information
    #     censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))   # Logging Information
    #     survtime_all = np.concatenate((survtime_all, survtime.detach().cpu().numpy().reshape(-1)))   # Logging Information
    #     case_all = np.concatenate((case_all, case.detach().cpu().numpy().reshape(-1)))   # Logging Information
    # df['pid'] = case_all
    # df['death'] = censor_all
    # df['time'] = survtime_all
    # df['risk_pred'] = risk_pred_all
    # df.to_csv('{}.csv'.format(args.fold))
