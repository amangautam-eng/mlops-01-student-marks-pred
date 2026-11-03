from src.pipeline.predict_pipeline import PredictPipeline
from src.pipeline.predict_pipeline import CustomData

test_data=CustomData(
    gender='male',
    race_ethnicity='group B',
    parental_level_of_education='some college',
    lunch='standard',
    test_preparation_course='completed',
    reading_score=50,
    writing_score=80)

test_df=test_data.get_data_as_frame()

pred=PredictPipeline()

print(pred.predict(test_df))
