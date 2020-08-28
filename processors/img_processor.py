import lmdb
import pickle

def img_features_loader(db_dir, img_id):
    """ Function that loads from the directory db_dir the fc6 features associated with the image with id img_id.
        
        Params:
            - db_dir: directory with the .lmdb files containing the image features
            - img_id: image id
        
        Returns:
            - img_feat: corresponding image features - format: np.array of shape (N=100, 2048) and type np.float32
    """

    # open lmdb environment
    env_db = lmdb.open(path=db_dir, subdir=True, readonly=True, readahead=False)

    # start transaction
    txn = env_db.begin()
    
    # convert img_id to proper byte string for keys in database
    id_str = str(img_id)
    if len(id_str) < 5:
        id_str = '0' + id_str
    assert len(id_str) == 5
    
    # retreive value in database (in pickle format) associated to id_str
    value = txn.get(id_str.encode()) 
    img_info = pickle.loads(value)

    # reteive image features only
    img_feat = img_info["features"]

    # close lmdb environment
    env_db.close()

    return img_feat