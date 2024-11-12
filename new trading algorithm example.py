#This is a version of my trading algorithm. At the moment it is quite simple, and although I am currently working on a more complicated version of it
#I am currently having trouble getting the buying power to work correctly(Quant Connect calculates option prices in quite a counter intuitive way)
#There are many things that I am looking forward to improving in the future and one of these things is in my AI module I am going to make a linear regression
#model to try and get a more accurate representation of implied volatility compared with realised volatility and then add this to my algorithm
#I am also currently exporing different ways of calculating implied and realised volatility rather than just using the options given to me by Quant Connect
#And VXX
#I thought you might want to see this simple version just to get a general idea of what I do, any advice would be greatly appreciated

#NOTE: this code will not work unless run through QuantConnect
#I have added lots of comments to help its readability 


# Import necessary libraries from QuantConnect
from AlgorithmImports import *
import numpy as np
from datetime import timedelta
from System import Decimal 

class VolatilityRiskPremiumWithHedgesAlgorithm(QCAlgorithm):
    
    def Initialize(self):

        # Set the start and end dates for backtesting
        self.SetStartDate(2010, 1, 1) 
        self.SetEndDate(2022, 12, 31)  
        
        # Set the initial cash for the algorithm
        self.SetCash(100000)            
        
        
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        
        
        self.vxx = self.AddEquity("VXX", Resolution.Daily).Symbol
        
        # Add SPY Options to the algorithm
        option = self.AddOption("SPY", Resolution.Daily)
        self.spy_option = option.Symbol
        self.option_contract = None  # To track the sold option contract
        
        # Define the lookback period for calculating moving averages and standard deviations
        self.lookback = 20  # 20 trading days
        
        # Initialize rolling window for SPY and VXX prices
        self.spy_window = RollingWindow[float](self.lookback)
        self.vxx_window = RollingWindow[float](self.lookback)
        
        # Define the option strategy parameters
        self.option_strike_offset = 2  
        self.option_expiry = 30        
        
        
        option.SetFilter(-1, 1, timedelta(days=1), timedelta(days=self.option_expiry + 30))
        
        # Schedule a daily event to rebalance the portfolio at market close
        self.Schedule.On(self.DateRules.EveryDay(self.spy),
                         self.TimeRules.BeforeMarketClose(self.spy, 15),
                         Action(self.Rebalance))
        
        # Schedule a daily event to sell hedges (options) shortly after rebalancing
        self.Schedule.On(self.DateRules.EveryDay(self.spy),
                         self.TimeRules.BeforeMarketClose(self.spy, 14),
                         Action(self.SellHedges))
        
        self.spy_holding = None
        self.vxx_holding = None
    
    def OnData(self, data):
        if self.spy in data:
            # Get the closing price of SPY
            try:
                spy_price = data[self.spy].Close
                # Add the price to the rolling window
                self.spy_window.Add(spy_price)
            except:
                pass
        
        if self.vxx in data:
            # Get the closing price of VXX
            try:
                vxx_price = data[self.vxx].Close
                # Add the price to the rolling window
                self.vxx_window.Add(vxx_price)
            except:
                pass
        
        if self.spy_option in data.OptionChains:
            chain = data.OptionChains[self.spy_option]
            # Select the first call option that is out-of-the-money and suitable for selling
            # Sort by expiration and strike price
            sorted_chain = sorted([x for x in chain if x.Right == OptionRight.Call and x.Strike > self.GetCurrentStrike() and x.Expiry > self.Time + timedelta(days=self.option_expiry)],
                                  key=lambda x: (x.Expiry, x.Strike))
            if sorted_chain:
                # Select the front month, out-of-the-money call option
                selected_option = sorted_chain[0]
                self.option_contract = selected_option.Symbol
    
    def GetCurrentStrike(self):
        if self.spy_window.IsReady:
            return self.spy_window[0]
        else:
            return self.Portfolio[self.spy].Price
    
    def Rebalance(self):

        # Ensure we have enough data to make a decision
        if not self.spy_window.IsReady or not self.vxx_window.IsReady:
            return
        
        # Calculate the average price of SPY and VXX over the lookback period
        spy_mean = np.mean([price for price in self.spy_window])
        vxx_mean = np.mean([price for price in self.vxx_window])
        
        # Calculate the standard deviation (vol) of SPY and VXX over the lookback period
        spy_std = np.std([price for price in self.spy_window])
        vxx_std = np.std([price for price in self.vxx_window])
        
        # Calculate the current price of SPY and VXX
        spy_current = self.spy_window[0]
        vxx_current = self.vxx_window[0]
        
        # Define thresholds for high and low volatility(needs to be tuned)
    
        high_vol_threshold = vxx_mean + vxx_std 
        low_vol_threshold = vxx_mean - vxx_std   
        
        
        self.Log(f"SPY Current: {spy_current:.2f}, SPY Mean: {spy_mean:.2f}, SPY STD: {spy_std:.2f}")
        self.Log(f"VXX Current: {vxx_current:.2f}, VXX Mean: {vxx_mean:.2f}, VXX STD: {vxx_std:.2f}")
        self.Log(f"High Vol Threshold: {high_vol_threshold:.2f}, Low Vol Threshold: {low_vol_threshold:.2f}")
        
        
        if vxx_current > high_vol_threshold:
            # High volatility detected
            # Strategy: Sell VXX and Buy SPY (expecting volatility to decrease)
            self.Log("High volatility detected. Selling VXX and Buying SPY.")
            
            allocation = 0.5
            
            # Liquidate existing VXX holdings if any
            if self.Portfolio[self.vxx].Invested:
                self.Liquidate(self.vxx)
                self.Log("Liquidated existing VXX holdings.")
            
            # Set SPY holdings to 50% of the portfolio
            self.SetHoldings(self.spy, allocation)
            self.Log(f"Set SPY holdings to {allocation * 100}% of portfolio.")
        
        elif vxx_current < low_vol_threshold:
            # Low volatility detected
            # Strategy: Buy VXX and Sell SPY (expecting volatility to increase)
            self.Log("Low volatility detected. Buying VXX and Selling SPY.")
            
            allocation = 0.5
            
            # Liquidate existing SPY holdings if any
            if self.Portfolio[self.spy].Invested:
                self.Liquidate(self.spy)
                self.Log("Liquidated existing SPY holdings.")
            
            # Set VXX holdings to 50% of the portfolio
            self.SetHoldings(self.vxx, allocation)
            self.Log(f"Set VXX holdings to {allocation * 100}% of portfolio.")
        
        else:
            # Volatility is within the normal range; maintain current positions
            self.Log("Volatility within normal range. Maintaining current positions.")
            pass  
    
    def SellHedges(self):
        if not self.spy_window.IsReady or not self.vxx_window.IsReady:
            return
        
        #the below code is quite similar to the above function so should be easy enough to understand
        spy_mean = np.mean([price for price in self.spy_window])
        vxx_mean = np.mean([price for price in self.vxx_window])
        
        spy_std = np.std([price for price in self.spy_window])
        vxx_std = np.std([price for price in self.vxx_window])
        
        spy_current = self.spy_window[0]
        vxx_current = self.vxx_window[0]
        

        high_vol_threshold = vxx_mean + vxx_std  
        low_vol_threshold = vxx_mean - vxx_std   
     
        if vxx_current < low_vol_threshold:
            self.Log("Low volatility confirmed for hedging. Selling SPY Call Options.")
            

            if self.option_contract is not None:

                if not self.Portfolio[self.option_contract].Invested:

                    contracts_to_sell = 1  # Adjust based on risk tolerance and portfolio size(should be adjusted)
                    
                    option_security = self.Securities[self.option_contract]
                    
                    option_bid_price = option_security.BidPrice
                    if option_bid_price is None or option_bid_price == 0:
                        self.Log("Option bid price not available or zero. Skipping selling options.")
                        return
                
                    total_premium = option_bid_price * 100 * contracts_to_sell 
                    
                    if self.Portfolio.Cash > total_premium:

                        self.Sell(self.option_contract, contracts_to_sell)
                        self.Log(f"Sold {contracts_to_sell} SPY Call Option(s) at {option_bid_price:.2f} each.")
                    else:
                        self.Log("Insufficient cash to sell SPY Call Options.")
                else:
                    self.Log("Already holding SPY Call Options. No action taken.")
            else:
                self.Log("No option contract selected to sell.")
        else:
            if self.option_contract is not None and self.Portfolio[self.option_contract].Invested:
                # Liquidate existing option positions
                self.Liquidate(self.option_contract)
                self.Log("Liquidated existing SPY Call Option(s) as volatility is no longer low.")
    
    def OnEndOfDay(self):

        # Ensure that both RollingWindows have enough data
        if self.spy_window.IsReady and self.vxx_window.IsReady:

            spy_mean = np.mean([price for price in self.spy_window])
            
            vxx_mean = np.mean([price for price in self.vxx_window])
            

            vxx_current = self.vxx_window[0]
            
            # Plot the calculated means and current VXX price
            self.Plot("Volatility Metrics", "SPY Mean", spy_mean)
            self.Plot("Volatility Metrics", "VXX Mean", vxx_mean)
            self.Plot("Volatility Metrics", "VXX Current", vxx_current)
        else:
            self.Log("RollingWindow not ready. Skipping end-of-day plots.")
    
    def OnOrderEvent(self, orderEvent):

        if orderEvent.Status == OrderStatus.Filled:
            self.Log(f"Order Filled: {orderEvent.Symbol.Value} - {orderEvent.Direction} {orderEvent.FillQuantity} @ {orderEvent.FillPrice}")
