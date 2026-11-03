from atlas_ai.config.settings import load_settings
from atlas_ai.execution.broker import HFMConnector
from atlas_ai.learning.historical_trainer import HistoricalTrainer
from atlas_ai.memory.knowledge_store import KnowledgeStore

print('trace:start', flush=True)
settings = load_settings()
print('trace:settings', settings.historical_train_max_bars, settings.historical_backtest_max_bars, settings.evolved_strategy_train_max_bars, flush=True)
store = KnowledgeStore(settings.storage_dir)
connector = HFMConnector(settings)
print('trace:connect', flush=True)
connector.connect()
print('trace:connected', flush=True)
trainer = HistoricalTrainer(settings, store)
print('trace:train', flush=True)
result = trainer.train()
print('trace:train_done', result, flush=True)
connector.shutdown()
print('trace:shutdown', flush=True)
