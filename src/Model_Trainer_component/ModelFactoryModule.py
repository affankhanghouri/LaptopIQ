from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


class ModelFactory:
    """
    Creates untrained model objects only
    """

    @staticmethod
    def get_model(model_name: str, params: dict):

        if model_name == "RandomForestRegressor":
            return RandomForestRegressor(**params)

        elif model_name == "LinearRegression":
            return LinearRegression(**params)

        else:
            raise ValueError(f"Unsupported model: {model_name}")
