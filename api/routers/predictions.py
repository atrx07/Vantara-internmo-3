"""Single-customer and canonical transaction batch scoring endpoints."""

from __future__ import annotations

import io
from datetime import UTC, datetime
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from sqlalchemy.orm import Session

from api.batch import prepare_batch_customers
from api.database import get_session
from api.models import Customer
from api.schemas.serving import PredictionRequest, PredictionResponse
from api.services import persist_score, prediction_payload
from src.data.validation import DataValidationError

router = APIRouter(tags=["predictions"])


@router.post("/predict/customer", response_model=PredictionResponse)
def predict_customer(
    payload: PredictionRequest,
    request: Request,
    session: Annotated[Session, Depends(get_session)],
) -> PredictionResponse:
    """Score and persist one known customer using startup-loaded artifacts."""
    customer = session.get(Customer, payload.customer_id)
    if customer is None:
        raise HTTPException(status_code=404, detail=f"Unknown customer_id: {payload.customer_id}")
    if payload.as_of_date is not None:
        supplied = payload.as_of_date
        supported = customer.feature_as_of
        if supplied.tzinfo is not None:
            supplied = supplied.astimezone(UTC).replace(tzinfo=None)
        if supported.tzinfo is not None:
            supported = supported.astimezone(UTC).replace(tzinfo=None)
        if supplied != supported:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Only the persisted as-of timestamp "
                    f"{customer.feature_as_of.isoformat()} is supported"
                ),
            )
    score = request.app.state.artifacts.score(customer.feature_payload, customer.sequence_payload)
    prediction = persist_score(session, customer, score, request.app.state.artifacts)
    return PredictionResponse(**prediction_payload(customer, score, prediction))


@router.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(
    request: Request,
    file: Annotated[UploadFile, File(description="Canonical transaction-level CSV")],
    session: Annotated[Session, Depends(get_session)],
) -> list[PredictionResponse]:
    """Validate transaction history, score every eligible customer, and persist results."""
    if file.content_type not in {"text/csv", "application/csv", "application/vnd.ms-excel"}:
        raise HTTPException(status_code=415, detail="Batch upload must be a CSV file")
    try:
        content = file.file.read(
            int(request.app.state.config["serving"]["batch_maximum_bytes"]) + 1
        )
        if len(content) > int(request.app.state.config["serving"]["batch_maximum_bytes"]):
            raise DataValidationError("Batch CSV exceeds the configured byte limit")
        frame = pd.read_csv(io.BytesIO(content))
        prepared = prepare_batch_customers(
            frame,
            registry=request.app.state.artifacts,
            config=request.app.state.config,
        )
    except (DataValidationError, pd.errors.ParserError, UnicodeDecodeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    if len(prepared) > int(request.app.state.config["serving"]["batch_maximum_customers"]):
        raise HTTPException(status_code=422, detail="Batch contains too many eligible customers")
    responses: list[PredictionResponse] = []
    for item in prepared:
        customer = session.get(Customer, item.customer_id)
        if customer is None:
            customer = Customer(
                customer_id=item.customer_id,
                country=item.country,
                feature_as_of=item.as_of.to_pydatetime(),
                feature_schema_version=(
                    f"{request.app.state.artifacts.freeze['feature_schema_version']}+"
                    f"{request.app.state.artifacts.clv['metadata']['feature_schema_version']}"
                ),
                feature_payload=item.features,
                sequence_payload=item.sequence,
                net_spend=float(item.features["net_spend"]),
                value_tier="unclassified",
                source_sha256=str(request.app.state.artifacts.freeze["source_sha256"]),
                loaded_at=datetime.now(UTC),
            )
            session.add(customer)
        else:
            customer.country = item.country
            customer.feature_as_of = item.as_of.to_pydatetime()
            customer.feature_payload = item.features
            customer.sequence_payload = item.sequence
            customer.net_spend = float(item.features["net_spend"])
        session.flush()
        score = request.app.state.artifacts.score(item.features, item.sequence)
        prediction = persist_score(session, customer, score, request.app.state.artifacts)
        responses.append(PredictionResponse(**prediction_payload(customer, score, prediction)))
    return responses
