from pydantic import BaseModel

class TransactionData(BaseModel):
    CurrencyCode: str
    CountryCode: int
    ProviderId: str
    ProductId: str
    ProductCategory: str
    ChannelId: str
    Amount: float
    Value: int
    PricingStrategy: int
    # TransactionHour: int
    # TransactionDay: int
    # TransactionMonth: int
    # TransactionYear: int