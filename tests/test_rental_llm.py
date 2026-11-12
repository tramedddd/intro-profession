import json
from pathlib import Path
import unittest

import pandas as pd
from pydantic import ValidationError
from rental_llm import RentalInfo, parse_count, evaluate_counts, evaluate_structured, structured_chain

ROOT = Path(__file__).resolve().parents[1]


class RentalTests(unittest.TestCase):
    def labels(self):
        return json.loads((ROOT/'rental_18_labels.json').read_text(encoding='utf-8'))

    def test_labels_match_dataset_and_schema(self):
        labels=self.labels()
        df=pd.read_csv(ROOT/'rental_18.csv',sep=';')
        self.assertEqual(len(labels),15)
        self.assertEqual(df['amount'].tolist(),[x['count'] for x in labels])
        for label in labels:
            data={k:v for k,v in label.items() if k!='id'}
            RentalInfo.model_validate(data)

    def test_no_fabricated_month_or_year(self):
        labels=self.labels()
        self.assertIsNone(labels[5]['start_date'])
        self.assertEqual(labels[2]['start_date'],'2018-05-29')
        self.assertEqual(labels[0]['start_date'],'--06-30')

    def test_invalid_date_and_counts_rejected(self):
        base={k:v for k,v in self.labels()[0].items() if k!='id'}
        for change in [{'start_date':'--02-30'},{'count':7},{'count':True},{'nights':-1}]:
            with self.assertRaises(ValidationError):
                RentalInfo.model_validate(base|change)

    def test_malformed_counts_not_silently_extracted(self):
        self.assertEqual(parse_count(' 4\n'),4)
        for raw in ['4 человека','3.0','0','-1','ошибка','3 или 4']:
            with self.assertRaises(ValueError):
                parse_count(raw)

    def test_errors_remain_in_denominator(self):
        metric=evaluate_counts(pd.DataFrame({'amount':[3,2,1],'result':['3','error','2']}))
        self.assertEqual(metric['correct'],1)
        self.assertEqual(metric['errors'],2)
        self.assertAlmostEqual(metric['accuracy'],1/3)

    def test_failed_json_does_not_match_null_label(self):
        labels=self.labels()
        records=[{'status':'ok','parsed':{k:v for k,v in x.items() if k!='id'}} for x in labels]
        records[0]={'status':'error','parsed':None}
        metrics=evaluate_structured(records,labels)
        self.assertEqual(metrics['fields']['price_per_day']['correct'],14)
        self.assertAlmostEqual(metrics['mean_extra_accuracy'],14/15)


if __name__ == '__main__':
    unittest.main()
