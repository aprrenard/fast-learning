import server_path

mouse_id = 'AR103'


data_folder = server_path.get_experimenter_analysis_folder('AR')
output_folder = os.path.join(output_folder, mouse_id, 'fissa')
if not os.path.join(output_folder):
    os.mkdir(output_folder)

bin_file = 

if not(os.path.isdir(fissaResultOutputPath)):

    #create the server paths

    binFile = os.path.join(serverDir, "analysis", "Georgios_Foustoukos", "RegisteredMovies", mouseName,"data.bin")
    tiffSavePath = os.path.join(serverDir, "analysis", "Georgios_Foustoukos", "FISSATifFiles", mouseName, mouseName + '_' + sessionDate)
    if 'AR' in mouseName:
        binFile = os.path.join('D:\\AR\\Foustoukos_et_al\\Suite2PRois', mouseName, "suite2p\\plane0\\data.bin")
        tiffSavePath = os.path.join('D:\\AR\\Foustoukos_et_al\\FISSATifFiles', mouseName, mouseName + '_' + sessionDate)

    if os.path.isdir(tiffSavePath):
        print('The tiff dir is already created')
    else:
        Path(tiffSavePath).mkdir(parents=True, exist_ok=True)
        print('The tiff dir has been created')

    keyDayToCalc = key.copy()

    dateToCalculate = (lsensGF.CalciumSession() & keyDayToCalc).fetch('session_date')

    keymouse =  {'mouse_name' : key['mouse_name']}

    CalciumDates = (lsensGF.CalciumSession() & keymouse).fetch('session_date')

    DaysBefore = [d for d in CalciumDates if d < dateToCalculate]

    framesToDay = 0

    for d in DaysBefore:
        keyDay = keymouse.copy()
        keyDay['session_date'] = d
        framesDays = np.sum((lsensGF.DataChunkTimes & keyDay).fetch('chunk_frames'))
        framesToDay += framesDays


    framesDayToCalc = np.sum((lsensGF.DataChunkTimes & keyDayToCalc).fetch('chunk_frames'))

    print(str(framesDayToCalc) + ' frames will be calculated..')

    Lx = int((lsensGF.CalciumSession() & key).fetch('fov_x'))
    Ly = int((lsensGF.CalciumSession() & key).fetch('fov_y'))

    nbytestoread = 2*Lx*Ly

    file = -1

    movie = np.zeros((1000, Ly, Lx), dtype = 'int16')

    with open(binFile, 'rb') as binary_file:

            for cframe in range(framesDayToCalc):

                bytesToSeek = int(nbytestoread*(framesToDay + cframe))
                binary_file.seek(bytesToSeek,0)
                buff = binary_file.read(nbytestoread)
                movie[np.mod(cframe,1000),:,:] = np.reshape(np.frombuffer(buff, dtype = np.int16, offset = 0),(Ly,Lx))

                if (np.mod(cframe,1000) == 999):

                    file += 1

                    fileLen = len(str(file))

                    zeros = str(0) * (4-fileLen) + str(file)

                    print('Doing frame: ' + str(cframe + 1))

                    finalPathToSave =  os.path.join(tiffSavePath, mouseName + '_' + sessionDate + '_' + zeros + '.tif')

                    with TiffWriter(finalPathToSave, bigtiff=True) as tif:
                        tif.save(movie, photometric='minisblack')

                    movie = np.zeros((1000, Ly, Lx), dtype = 'int16')

            file += 1

            fileLen = len(str(file))

            zeros = str(0) * (4-fileLen) + str(file)

            print('Doing frame: ' + str(cframe + 1))

            finalPathToSave =  os.path.join(tiffSavePath, mouseName + '_' + sessionDate + '_' + zeros + '.tif')

            lastFrames = framesDayToCalc - (file + 1)*1000

            movieToEnd = movie[0:lastFrames,:,:]

            with TiffWriter(finalPathToSave, bigtiff=True) as tif:
                tif.save(movieToEnd, photometric='minisblack')

            print('Tiff writing has finished..')

            # Start the FISSA Computation

    # Extract the motion corrected tiffs (make sure that the reg_tif option is set to true, see above)
    images = tiffSavePath

    # Load the detected regions of interest
    if 'GF' in mouseName or 'MI' in mouseName:
        suite2pRoisPath = os.path.join(serverDir, "analysis","Georgios_Foustoukos","Suite2PRois", mouseName)
        stat = np.load(os.path.join(serverDir, "analysis","Georgios_Foustoukos","Suite2PRois", mouseName, 'suite2p', 'plane0', 'stat.npy'), allow_pickle = True)
        ops = np.load(os.path.join(serverDir, "analysis","Georgios_Foustoukos","Suite2PRois", mouseName, 'suite2p', 'plane0', 'ops.npy'), allow_pickle = True).item()
        iscell = np.load(os.path.join(serverDir, "analysis","Georgios_Foustoukos","Suite2PRois", mouseName, 'suite2p', 'plane0', 'iscell.npy'), allow_pickle = True)[:,0]
    if 'AR' in mouseName:
        suite2pRoisPath = os.path.join('D:\\AR\\Foustoukos_et_al\\Suite2PRois', mouseName, 'suite2p', 'plane0')
        stat = np.load(os.path.join(suite2pRoisPath,'stat.npy'), allow_pickle = True)
        ops = np.load(os.path.join(suite2pRoisPath,'ops.npy'), allow_pickle = True).item()
        iscell = np.load(os.path.join(suite2pRoisPath,'iscell.npy'), allow_pickle = True)[:,0]

    print('Suite2p files loaded..')

    if 'inmerge' in stat[0].keys():
        for i in range(len(stat)):

            if not((stat[i]['inmerge'] == 0.0) or (stat[i]['inmerge'] == -1.0)):

                iscell[i] = 0.0

        print('Merged ROIs corrected after merging..')
    else:
        print('No ROIs merge in this session.')

    # Get image size
    Lx = ops['Lx']
    Ly = ops['Ly']

    # Get the cell ids
    ncells = len(stat)
    cell_ids = np.arange(ncells)  # assign each cell an ID, starting from 0.
    cell_ids = cell_ids[iscell==1]  # only take the ROIs that are actually cells.
    num_rois = len(cell_ids)

    # Generate ROI masks in a format usable by FISSA (in this case, a list of masks)
    rois = [np.zeros((Ly, Lx), dtype=bool) for n in range(num_rois)]

    for i, n in enumerate(cell_ids):

        # i is the position in cell_ids, and n is the actual cell number
        if 'imerge' in stat[0].keys():
            if np.array(stat[n]['imerge']).any():
                ypix = stat[n]['ypix']
                xpix = stat[n]['xpix']
            else:
                ypix = stat[n]['ypix'][~stat[n]['overlap']]
                xpix = stat[n]['xpix'][~stat[n]['overlap']]
        else:
            ypix = stat[n]['ypix'][~stat[n]['overlap']]
            xpix = stat[n]['xpix'][~stat[n]['overlap']]

        if (np.sum(xpix) == 0 or np.sum(ypix) == 0):
            print(n)

        rois[i][ypix, xpix] = 1

    fissaOutputPath = os.path.join(serverDir, "analysis","Georgios_Foustoukos","FISSAOutputFiles",mouseName, mouseName + '_' + sessionDate)
    if 'AR' in mouseName:
        fissaOutputPath = os.path.join('D:\\AR\\Foustoukos_et_al\\FISSAOutputFiles', mouseName, mouseName + '_' + sessionDate)

    if os.path.isdir(fissaOutputPath ):
        print('The fissa  dir is already created..')
    else:
        Path(fissaOutputPath).mkdir(parents=True, exist_ok=True)
        print('The fissa dir has been created')

    print('Start running FISSA..')

    exp = fissa.Experiment(images, [rois[:ncells]], fissaOutputPath)
    exp.separate()

    # save the final F signal after FISSA computation

    fissaResultOutputPath = os.path.join(serverDir, "analysis","Georgios_Foustoukos","FISSASessionData",mouseName, mouseName + '_' + sessionDate)
    if 'AR' in mouseName:
        fissaResultOutputPath = os.path.join('D:\\AR\\Foustoukos_et_al\\FISSASessionData',mouseName, mouseName + '_' + sessionDate)

    if os.path.isdir(fissaResultOutputPath):
        print('The fissa final dir is already created..')
    else:
        Path(fissaResultOutputPath).mkdir(parents=True, exist_ok=True)
        print('The fissa final dir has been created..')

    print('Computing final F signal for ' + str(num_rois) + ' rois and ' + str(framesDayToCalc) + ' frames, split in ' + str(file+1) + ' files.')

    F = np.empty((num_rois,framesDayToCalc))

    try:
        print('Old fissa file')
        results = np.load(os.path.join(fissaOutputPath, 'separated.npy'), allow_pickle = True)
        Fmatched = results[3]

    except:
        print('New fissa file')
        results = np.load(os.path.join(fissaOutputPath, 'separated.npz'), allow_pickle = True)
        Fmatched = results['result']

    for r in range(num_rois):

        #print('Doing roi: ' + str(r))
        Temp = np.array([])

        for t in range((file + 1)):
            FTrial = Fmatched[r][t][0,:]
            #print('Doing chunck: ' + str(t) + ', frames: ' + str(FTrial.shape[0]))
            Temp = np.concatenate((Temp,Fmatched[r][t][0,:]))

        F[r,:] = Temp

    np.save(os.path.join(fissaResultOutputPath,"F_fissa.npy"),F)

    print('Final F signal from FISSA saved..')

    print('The computation for this session is now complete, the tif files will be deleted...')
    # Remove the Directory

    shutil.rmtree(tiffSavePath)

    key['fissasavepath'] = fissaResultOutputPath

    self.insert1(key)
else:
    print('The session is already computed..')
    key['fissasavepath'] = fissaResultOutputPath
