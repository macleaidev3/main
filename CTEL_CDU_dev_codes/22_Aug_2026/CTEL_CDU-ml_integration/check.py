import joblib

path = r"D:\Anurag BPCL WORK\SENTINAL SOFTWARE\CTEL_CDU-ml_integration\ml_module\cache_mlmodule_1232232\ID_00001\target_scaler.pkl"

try:
    obj = joblib.load(path)
    print("SUCCESS")
    print(type(obj))
except Exception as e:
    print("FAILED")
    print(type(e).__name__)
    print(str(e))