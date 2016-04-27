import dataset.mcam_image.base
import dataset.mcam_image.mslweb_image_index
import dataset.mcam_image.udr_blacklist


class McamImage(dataset.mcam_image.base.BaseMcamImage):

    data_dir = '/molokini_raid/MSL/data/surface/processed/images/web/full/SURFACE/'
    save_dir = '/home/hannah/data/mcam-artifacts/saved/'
    tmp_dir = '/home/hannah/data/mcam-artifacts/tmp/'

    #### Query to generate mslweb_image_index.sql_query_results
    #### Retrieves McamRecoveredProducts with the sol of their corresponding McamImage
    # SELECT
    #     *
    # FROM
    #     (select
    #         name, udr_image_id
    #     from
    #         udr
    #     where
    #         type_of_product = 'RecoveredProduct'
    #             and (instrument = 'ML' or instrument = 'MR')
    #             and udr.name not like '%Partial%'
    #             and comp_quality = 0
    #             and udr.udr_image_id is NOT NULL
    #             and sol != 1000
    #             and width = 168 and height = 150) AS t1
    # NATURAL JOIN
    #     (select
    #         udr_image_id, sol
    #     from
    #         udr
    #     where
    #         type_of_product = 'Image'
    #             and (instrument = 'ML' or instrument = 'MR')
    #             and udr.name not like '%Partial%'
    #             and udr.udr_image_id is NOT NULL
    #             and sol != 1000
    #             and width = 168 and height = 150) AS t2

    _index = {
        name: sol
        for (udr, name, sol) in dataset.mcam_image.mslweb_image_index.sql_query_results
        if udr not in dataset.mcam_image.udr_blacklist.blacklist
    }

    all_names = _index.keys()

    @property
    def instrument(self):
        if 'McamL' in self.name:
            return 'ML'
        elif 'McamR' in self.name:
            return 'MR'

    @property
    def sol(self):
        return self._index[self.name]

    @property
    def _raw_image_path(self):
        return self.data_dir + '%s/%s/%s' % (
            self.instrument,
            'sol' + str(self.sol).zfill(4),
            self.name.strip('\"') + '.png'
        )