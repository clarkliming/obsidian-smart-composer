import OpenAI, { AzureOpenAI } from 'openai'

import { ChatModel } from '../../types/chat-model.types'
import {
  LLMOptions,
  LLMRequestNonStreaming,
  LLMRequestStreaming,
} from '../../types/llm/request'
import {
  LLMResponseNonStreaming,
  LLMResponseStreaming,
} from '../../types/llm/response'
import { LLMProvider } from '../../types/provider.types'

import {
  LLMAPIKeyInvalidException,
  LLMAPIKeyNotSetException,
  LLMRateLimitExceededException,
} from './exception'

import { BaseLLMProvider } from './base'
import { OpenAIMessageAdapter } from './openaiMessageAdapter'

export class AzureOpenAIProvider extends BaseLLMProvider<
  Extract<LLMProvider, { type: 'azure-openai' }>
> {
  private adapter: OpenAIMessageAdapter
  private client: OpenAI

  constructor(provider: Extract<LLMProvider, { type: 'azure-openai' }>) {
    super(provider)
    this.adapter = new OpenAIMessageAdapter()
    this.client = new AzureOpenAI({
      apiKey: provider.apiKey ?? '',
      endpoint: provider.baseUrl ?? '',
      apiVersion: provider.additionalSettings.apiVersion,
      deployment: provider.additionalSettings.deployment,
      dangerouslyAllowBrowser: true,
    })
  }

  async generateResponse(
    model: ChatModel,
    request: LLMRequestNonStreaming,
    options?: LLMOptions,
  ): Promise<LLMResponseNonStreaming> {
    if (model.providerType !== 'azure-openai') {
      throw new Error('Model is not an Azure OpenAI model')
    }

    return this.adapter.generateResponse(this.client, request, options)
  }

  async streamResponse(
    model: ChatModel,
    request: LLMRequestStreaming,
    options?: LLMOptions,
  ): Promise<AsyncIterable<LLMResponseStreaming>> {
    if (model.providerType !== 'azure-openai') {
      throw new Error('Model is not an Azure OpenAI model')
    }

    return this.adapter.streamResponse(this.client, request, options)
  }

  async getEmbedding(model: string, text: string): Promise<number[]> {
    if (!this.client.apiKey) {
      throw new LLMAPIKeyNotSetException(
        `Provider ${this.provider.id} API key is missing. Please set it in settings menu.`,
      )
    }
    try {
      const embedding = await this.client.embeddings.create({
        model: model,
        input: text,
      })
      return embedding.data[0].embedding
    } catch (error) {
      if (error.status === 429) {
        throw new LLMRateLimitExceededException(
          'Azure OpenAI API rate limit exceeded. Please try again later.',
        )
      }
      throw error
    }
  }
}
