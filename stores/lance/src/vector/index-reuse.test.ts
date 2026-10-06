import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { Index } from '@lancedb/lancedb';
import type { Connection, Table } from '@lancedb/lancedb';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { LanceVectorStore } from './index';

describe('vector index configuration reuse', () => {
  let path: string;
  let store: LanceVectorStore;
  let client: Connection;
  let table: Table;
  const params = {
    tableName: 'documents',
    indexName: 'vector',
    dimension: 16,
    metric: 'euclidean' as const,
    indexConfig: { type: 'hnsw' as const, numPartitions: 1, numSubVectors: 2, hnsw: { m: 16, efConstruction: 100 } },
  };
  beforeEach(async () => {
    path = await mkdtemp(join(tmpdir(), 'mastra-lance-reuse-'));
    store = await LanceVectorStore.create(path);
    client = (store as unknown as { lanceClient: Connection }).lanceClient;
    table = await client.createTable(
      'documents',
      Array.from({ length: 300 }, (_, i) => ({
        id: String(i),
        vector: Array.from({ length: 16 }, (_, j) => Math.sin(i * 17 + j + 1)),
      })),
    );
    vi.spyOn(client, 'openTable').mockResolvedValue(table);
  });
  afterEach(async () => {
    vi.restoreAllMocks();
    store.close();
    await rm(path, { recursive: true, force: true });
  });
  it('reuses the same build configuration', async () => {
    await store.createIndex(params);
    const createIndex = vi.spyOn(table, 'createIndex');
    await store.createIndex(params);
    expect(createIndex).not.toHaveBeenCalled();
  });
  it.each([
    ['m', { ...params.indexConfig, hnsw: { ...params.indexConfig.hnsw, m: 24 } }],
    ['efConstruction', { ...params.indexConfig, hnsw: { ...params.indexConfig.hnsw, efConstruction: 120 } }],
    ['numPartitions', { ...params.indexConfig, numPartitions: 2 }],
    ['numSubVectors', { ...params.indexConfig, numSubVectors: 4 }],
  ])('rebuilds when %s changes', async (_name, indexConfig) => {
    await store.createIndex(params);
    const createIndex = vi.spyOn(table, 'createIndex');
    await store.createIndex({ ...params, indexConfig });
    expect(createIndex).toHaveBeenCalledOnce();
    await store.createIndex({ ...params, indexConfig });
    expect(createIndex).toHaveBeenCalledOnce();
  });
  it('reuses an index after reopening the store', async () => {
    await store.createIndex(params);
    store.close();
    store = await LanceVectorStore.create(path);
    client = (store as unknown as { lanceClient: Connection }).lanceClient;
    table = await client.openTable('documents');
    vi.spyOn(client, 'openTable').mockResolvedValue(table);
    const createIndex = vi.spyOn(table, 'createIndex');
    await store.createIndex(params);
    expect(createIndex).not.toHaveBeenCalled();
  });
  it('rebuilds an index whose configuration was not recorded', async () => {
    await table.createIndex('vector', { config: Index.hnswPq({ m: 24, efConstruction: 120, distanceType: 'l2' }) });
    const createIndex = vi.spyOn(table, 'createIndex');
    await store.createIndex(params);
    expect(createIndex).toHaveBeenCalledOnce();
  });
  it('rebuilds after another client replaces the index', async () => {
    await store.createIndex(params);
    await table.createIndex('vector', { config: Index.hnswPq({ m: 24, efConstruction: 120, distanceType: 'l2' }) });
    const createIndex = vi.spyOn(table, 'createIndex');
    await store.createIndex(params);
    expect(createIndex).toHaveBeenCalledOnce();
  });
  it.each([
    ['numPartitions', { type: 'ivfflat' as const, numPartitions: 2, numSubVectors: 2 }],
    ['numSubVectors', { type: 'ivfflat' as const, numPartitions: 1, numSubVectors: 4 }],
  ])('rebuilds IVF PQ when %s changes', async (_name, indexConfig) => {
    const initial = { ...params, indexConfig: { type: 'ivfflat' as const, numPartitions: 1, numSubVectors: 2 } };
    await store.createIndex(initial);
    const createIndex = vi.spyOn(table, 'createIndex');
    await store.createIndex(initial);
    expect(createIndex).not.toHaveBeenCalled();
    await store.createIndex({ ...params, indexConfig });
    expect(createIndex).toHaveBeenCalledOnce();
    await store.createIndex({ ...params, indexConfig });
    expect(createIndex).toHaveBeenCalledOnce();
  });

  it('rebuilds when the metric or index type changes', async () => {
    await store.createIndex(params);
    const createIndex = vi.spyOn(table, 'createIndex');
    await store.createIndex({ ...params, metric: 'cosine' });
    expect(createIndex).toHaveBeenCalledOnce();
    await store.createIndex({ ...params, indexConfig: { type: 'ivfflat', numPartitions: 1, numSubVectors: 2 } });
    expect(createIndex).toHaveBeenCalledTimes(2);
  });

  it('normalizes omitted HNSW defaults and forwards partition and compression settings', async () => {
    const buildIndex = vi.spyOn(Index, 'hnswPq');
    await store.createIndex({ ...params, indexConfig: { numPartitions: 1, numSubVectors: 2 } });
    expect(buildIndex).toHaveBeenCalledWith({
      m: 16,
      efConstruction: 100,
      numPartitions: 1,
      numSubVectors: 2,
      distanceType: 'l2',
    });
    const createIndex = vi.spyOn(table, 'createIndex');
    await store.createIndex(params);
    expect(createIndex).not.toHaveBeenCalled();
    expect((await table.listIndices())[0]?.indexDetails).toMatchObject({
      hnsw: { max_connections: 16, construction_ef: 100 },
      compression: { num_sub_vectors: 2 },
    });
  });

  it('rebuilds rather than trusting malformed configuration metadata', async () => {
    await store.createIndex(params);
    await table.updateFieldMetadata([{ path: 'vector', metadata: { 'mastra.vector.indexConfig': 'invalid json' } }]);
    const createIndex = vi.spyOn(table, 'createIndex');
    await store.createIndex(params);
    expect(createIndex).toHaveBeenCalledOnce();
  });

  it('does not reuse an index without a physical identity', async () => {
    await store.createIndex(params);
    const listIndices = table.listIndices.bind(table);
    vi.spyOn(table, 'listIndices').mockImplementation(async () =>
      (await listIndices()).map(index => ({ ...index, indexUuid: undefined })),
    );
    const createIndex = vi.spyOn(table, 'createIndex');
    const updateMetadata = vi.spyOn(table, 'updateFieldMetadata');
    await store.createIndex(params);
    expect(createIndex).toHaveBeenCalledOnce();
    expect(updateMetadata).not.toHaveBeenCalled();
  });

  it('preserves unrelated column metadata', async () => {
    await table.updateFieldMetadata([{ path: 'vector', metadata: { application: 'keep' } }]);
    await store.createIndex(params);
    expect((await table.schema()).fields.find(field => field.name === 'vector')?.metadata.get('application')).toBe(
      'keep',
    );
  });

  it('keeps appended rows searchable without rebuilding the index', async () => {
    await store.createIndex(params);
    const vector = Array.from({ length: 16 }, (_, j) => 50 + j);
    await table.add([{ id: 'appended', vector }]);
    const createIndex = vi.spyOn(table, 'createIndex');
    await store.createIndex(params);
    expect(createIndex).not.toHaveBeenCalled();
    expect((await store.query({ indexName: 'documents', queryVector: vector, topK: 1 }))[0]?.id).toBe('appended');
    const index = (await table.listIndices())[0]!;
    expect((await table.indexStats(index.name))?.numUnindexedRows).toBe(1);
  });
  it('does not associate an unrelated named index with the managed configuration', async () => {
    await table.createIndex('vector', {
      name: 'external_index',
      config: Index.hnswPq({ m: 24, efConstruction: 120, distanceType: 'l2' }),
    });
    await store.createIndex(params);
    const createIndex = vi.spyOn(table, 'createIndex');
    await store.createIndex(params);
    expect(createIndex).not.toHaveBeenCalled();
    expect((await table.listIndices()).map(index => index.name)).toEqual(
      expect.arrayContaining(['external_index', 'vector_idx']),
    );
  });
});
