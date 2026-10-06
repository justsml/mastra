# @mastra/lance

`@mastra/lance` provides Mastra storage and vector search backed by the embedded LanceDB database. Use it when you want local, file-based persistence and semantic search without operating a separate database service.

## Installation

```bash
npm install @mastra/lance
```

## Usage

Create a local LanceDB vector store, then create an index before writing embeddings.

```typescript
import { LanceVectorStore } from '@mastra/lance';

const vectorStore = await LanceVectorStore.create('./data/lancedb');

await vectorStore.createIndex({
  indexName: 'documents',
  dimension: 1536,
});
```

Repeated `createIndex()` calls reuse an index only when its column, type, distance metric, and build settings match a configuration recorded by Mastra for that physical index. Build settings include `hnsw.m`, `hnsw.efConstruction`, `numPartitions`, and `numSubVectors`. The latter two settings are passed to both index builders. Omitted HNSW settings retain LanceDB's defaults for partitioning and compression, and Mastra's defaults of `m: 16` and `efConstruction: 100`.

An existing index without recorded configuration is rebuilt once. The configuration is stored in vector-column metadata and tied to the index UUID, so it remains usable after reopening the database and is invalidated if another client replaces the index. Backends that do not expose an index UUID rebuild conservatively.

Index reuse does not refresh the index with newly written rows. LanceDB normally searches unindexed rows through a fallback scan; index maintenance with `table.optimize()` is a separate operation. See [LanceDB's index management guidance](https://docs.lancedb.com/indexing/vector-index#managing-vector-indexes).

## Documentation

- [LanceDB integration guide](https://mastra.ai/integrations/databases/lancedb)
- [Lance vector reference](https://mastra.ai/reference/vectors/lance)

## Changelog

See the [package changelog](https://github.com/mastra-ai/mastra/blob/main/stores/lance/CHANGELOG.md) for version history and release notes.

## Support

We have an [open community Discord](https://discord.gg/mastra-ai). Come and say hello and let us know if you have any questions or need any help getting things running.
