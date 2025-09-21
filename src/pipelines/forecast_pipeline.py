"""
Pipeline to orchestrate the forecasting process.
"""




class ForecastPipeline:
    def __init__(self, config):
        self.config = config
        # TODO: initialize agents based on config

    async def run(self, question):
        """
        Run full forecast for a given question.
        """
        # TODO: select appropriate agent and run forecast
        # Example placeholder:
        # return await agent.forecast(question)
        raise NotImplementedError("ForecastPipeline.run() not implemented yet")
