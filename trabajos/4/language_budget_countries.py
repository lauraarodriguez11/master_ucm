from mrjob.job import MRJob

def to_int(value):
    """
    Safely convert a value to int. Return -1 if conversion fails.
    """
    try:
        return int(value)
    except ValueError:
        return -1
    
class LanguageBudgetCountries(MRJob):
    
    def mapper(self, _, line):
        """
        Mapper: Extracts relevant fields for each movie.
        Only considers rows where language, country, and budget are non-empty.
        """
        fields = line.split('|')
        if len(fields) < 5:
            return
        title, year, language, country, budget = fields

        if not language.strip() or not country.strip() or to_int(budget) == -1:
            return

        yield language.strip(), (country.strip(), to_int(budget))

    def reducer(self, key, values):
        """
        Reducer: Aggregates the budget for each language, with countries listed.
        """
        country_budgets = {}
        for country, budget in values:
            if country not in country_budgets:
                country_budgets[country] = 0
            country_budgets[country] += budget

        total_budget = sum(country_budgets.values())
        countries = list(country_budgets.keys())
        yield key, (countries, total_budget)

if __name__ == '__main__':
    LanguageBudgetCountries.run()