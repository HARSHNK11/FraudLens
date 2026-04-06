from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class TransactionInput(BaseModel):
    Time: float = Field(..., description="Seconds elapsed since the first transaction in the dataset")
    V1: Optional[float] = None; V2: Optional[float] = None; V3: Optional[float] = None; V4: Optional[float] = None; V5: Optional[float] = None
    V6: Optional[float] = None; V7: Optional[float] = None; V8: Optional[float] = None; V9: Optional[float] = None; V10: Optional[float] = None
    V11: Optional[float] = None; V12: Optional[float] = None; V13: Optional[float] = None; V14: Optional[float] = None; V15: Optional[float] = None
    V16: Optional[float] = None; V17: Optional[float] = None; V18: Optional[float] = None; V19: Optional[float] = None; V20: Optional[float] = None
    V21: Optional[float] = None; V22: Optional[float] = None; V23: Optional[float] = None; V24: Optional[float] = None; V25: Optional[float] = None
    V26: Optional[float] = None; V27: Optional[float] = None; V28: Optional[float] = None
    Amount: float = Field(..., ge=0, description="Transaction amount in USD")

    model_config = {
        "json_schema_extra": {
            "example": {
                "Time": 0.0,
                "V1": -1.3598071336738, "V2": -0.0727811733098497,
                "V3": 2.53634673796914, "V4": 1.37815522427443,
                "V5": -0.338320769942518, "V6": 0.462387777762292,
                "V7": 0.239598554061257, "V8": 0.0986979012610507,
                "V9": 0.363786969611213, "V10": 0.0907941719789513,
                "V11": -0.551599533260813, "V12": -0.617800855762348,
                "V13": -0.991389847235408, "V14": -0.311169353699879,
                "V15": 1.46817697209427, "V16": -0.470400525259478,
                "V17": 0.207971241929242, "V18": 0.0257905801985591,
                "V19": 0.403992960255733, "V20": 0.251412098239705,
                "V21": -0.018306777944153, "V22": 0.277837575558899,
                "V23": -0.110473910188767, "V24": 0.0669280749146731,
                "V25": 0.128539358273528, "V26": -0.189114843888824,
                "V27": 0.133558376740387, "V28": -0.0210530534538215,
                "Amount": 149.62
            }
        }
    }


class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    threshold_used: float
    shap_values: Optional[Dict[str, float]] = None
    top_features: Optional[list] = None


class ModelInfoResponse(BaseModel):
    model_name: str
    threshold: float
    metrics: Dict[str, Any]
    features: list[str]


class BatchTransactionInput(BaseModel):
    transactions: list[TransactionInput]


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    total: int
    fraud_count: int
