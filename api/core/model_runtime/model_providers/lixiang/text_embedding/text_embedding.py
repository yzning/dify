import time
from json import dumps
from typing import Optional

from requests import post
from requests.exceptions import ConnectionError, InvalidSchema, MissingSchema

from core.model_runtime.entities.common_entities import I18nObject
from core.model_runtime.entities.model_entities import (
    AIModelEntity,
    FetchFrom,
    ModelPropertyKey,
    ModelType,
    PriceType,
)
from core.model_runtime.entities.text_embedding_entities import EmbeddingUsage, TextEmbeddingResult
from core.model_runtime.errors.invoke import (
    InvokeAuthorizationError,
    InvokeBadRequestError,
    InvokeConnectionError,
    InvokeError,
    InvokeRateLimitError,
    InvokeServerUnavailableError,
)
from core.model_runtime.errors.validate import CredentialsValidateFailedError
from core.model_runtime.model_providers.__base.text_embedding_model import TextEmbeddingModel


class LixiangTextEmbeddingModel(TextEmbeddingModel):
    """
    Model class for OpenLLM text embedding model.
    """
    def _invoke(self, model: str, credentials: dict,
                texts: list[str], user: Optional[str] = None) \
            -> TextEmbeddingResult:
        """
        Invoke text embedding model

        :param model: model name
        :param credentials: model credentials
        :param texts: texts to embed
        :param user: unique user id
        :return: embeddings result
        """
        server_url = credentials['api_base_url']
        if not server_url:
            raise CredentialsValidateFailedError('server_url is required')

        headers = {
            'Content-Type': 'text/plain',
        }

        url = server_url
        embeddings = []
        total_tokens = self.get_num_tokens(model, credentials, texts)

        for text in texts:
            try:
                response = post(url, headers=headers, data=dumps({'query': text}))
            except (ConnectionError, InvalidSchema, MissingSchema) as e:
                # cloud not connect to the server
                raise InvokeAuthorizationError(f"Invalid server URL: {e}")
            except Exception as e:
                raise InvokeConnectionError(str(e))

            if response.status_code != 200:
                if response.status_code == 400:
                    raise InvokeBadRequestError(response.text)
                elif response.status_code == 404:
                    raise InvokeAuthorizationError(response.text)
                elif response.status_code == 500:
                    raise InvokeServerUnavailableError(response.text)

            try:
                resp = response.json()
                embedding = resp['embedding_result']
                embeddings.append(embedding)
            except KeyError as e:
                raise InvokeServerUnavailableError(f"Failed to convert response to json: {e} with text: {response.text}")

        usage = self._calc_response_usage(model=model, credentials=credentials, tokens=total_tokens)

        result = TextEmbeddingResult(
            model=model,
            embeddings=embeddings,
            usage=usage
        )

        return result

    def get_num_tokens(self, model: str, credentials: dict, texts: list[str]) -> int:
        """
        Get number of tokens for given prompt messages

        :param model: model name
        :param credentials: model credentials
        :param texts: texts to embed
        :return:
        """
        num_tokens = 0
        for text in texts:
            # use GPT2Tokenizer to get num tokens
            num_tokens += self._get_num_tokens_by_gpt2(text)
        return num_tokens


    def _get_customizable_model_schema(self, model: str, credentials: dict) -> AIModelEntity | None:
        """
        Get customizable model schema

        :param model: model name
        :param credentials: model credentials
        :return: model schema
        """
        return AIModelEntity(
            model=model,
            label=I18nObject(zh_Hans=model, en_US=model),
            model_type=ModelType.TEXT_EMBEDDING,
            features=[],
            fetch_from=FetchFrom.CUSTOMIZABLE_MODEL,
            model_properties={
                ModelPropertyKey.CONTEXT_SIZE: int(credentials.get('context_size', '512')),
                ModelPropertyKey.MAX_CHUNKS: 1,
            },
            parameter_rules=[]
        )

    def validate_credentials(self, model: str, credentials: dict) -> None:
        """
        Validate model credentials

        :param model: model name
        :param credentials: model credentials
        :return:
        """
        try:
            self._invoke(model=model, credentials=credentials, texts=['ping'])
        except InvokeAuthorizationError:
            raise CredentialsValidateFailedError('Invalid server_url')

    @property
    def _invoke_error_mapping(self) -> dict[type[InvokeError], list[type[Exception]]]:
        """
        Map model invoke error to unified error
        The key is the error type thrown to the caller
        The value is the error type thrown by the model,
        which needs to be converted into a unified error type for the caller.

        :return: Invoke error mapping
        """
        return {
            InvokeConnectionError: [
                InvokeConnectionError
            ],
            InvokeServerUnavailableError: [
                InvokeServerUnavailableError
            ],
            InvokeRateLimitError: [
                InvokeRateLimitError
            ],
            InvokeAuthorizationError: [
                InvokeAuthorizationError
            ],
            InvokeBadRequestError: [
                KeyError
            ]
        }

    def _calc_response_usage(self, model: str, credentials: dict, tokens: int) -> EmbeddingUsage:
        """
        Calculate response usage

        :param model: model name
        :param credentials: model credentials
        :param tokens: input tokens
        :return: usage
        """
        # get input price info
        input_price_info = self.get_price(
            model=model,
            credentials=credentials,
            price_type=PriceType.INPUT,
            tokens=tokens
        )

        # transform usage
        usage = EmbeddingUsage(
            tokens=tokens,
            total_tokens=tokens,
            unit_price=input_price_info.unit_price,
            price_unit=input_price_info.unit,
            total_price=input_price_info.total_amount,
            currency=input_price_info.currency,
            latency=time.perf_counter() - self.started_at
        )

        return usage
